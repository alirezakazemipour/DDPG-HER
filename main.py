import gym
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
from play import Play
import mujoco_py
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch

ENV_NAME = "FetchPickAndPlace-v1"
INTRO = False
MAX_EPOCHS = 50
MAX_CYCLES = 50
num_updates = 40
MAX_EPISODES = 16 // os.cpu_count()
memory_size = 9000 // 2  # 7e+5 // os.cpu_count()
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05
k_future = 4

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'

def eval_agent(env, agent):
    total_success_rate = []
    running_reward = []
    for ep in range(10):
        per_success_rate = []
        env_dict = env.reset()
        # agent.reset_randomness()
        s = env_dict["observation"]
        ag = env_dict["achieved_goal"]
        g = env_dict["desired_goal"]
        while np.linalg.norm(ag - g) <= 0.05:
            env_dict = env.reset()
            # agent.reset_randomness()
            s = env_dict["observation"]
            ag = env_dict["achieved_goal"]
            g = env_dict["desired_goal"]
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                action = agent.choose_action(s, g, train_mode=False)
            observation_new, reward, done, info = env.step(action)
            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info['is_success'])
            episode_reward += reward
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_reward.append(episode_reward)
        else:
            running_reward.append(running_reward[-1] * 0.99 + 0.01 * episode_reward)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_reward, episode_reward


if INTRO:
    print(f"state_shape:{state_shape}\n"
          f"number of actions:{n_actions}\n"
          f"action boundaries:{action_bounds}\n"
          f"max timesteps:{test_env._max_episode_steps}")
    for _ in range(3):
        done = False
        test_env.reset()
        while not done:
            action = test_env.action_space.sample()
            test_state, test_reward, test_done, test_info = test_env.step(action)
            # substitute_goal = test_state["achieved_goal"].copy()
            # substitute_reward = test_env.compute_reward(
            #     test_state["achieved_goal"], substitute_goal, test_info)
            # print("r is {}, substitute_reward is {}".format(r, substitute_reward))
            test_env.render()
else:
    env = gym.make(ENV_NAME)
    env.seed(MPI.COMM_WORLD.Get_rank())
    random.seed(MPI.COMM_WORLD.Get_rank())
    np.random.seed(MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(MPI.COMM_WORLD.Get_rank())
    agent = Agent(n_states=state_shape,
                  n_actions=n_actions,
                  n_goals=n_goals,
                  action_bounds=action_bounds,
                  capacity=memory_size,
                  action_size=n_actions,
                  batch_size=batch_size,
                  actor_lr=actor_lr,
                  critic_lr=critic_lr,
                  gamma=gamma,
                  tau=tau,
                  k_future=k_future,
                  env=dc(env))

    total_success_rate = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        for cycle in range(MAX_CYCLES):
            mb = []
            for episode in range(MAX_EPISODES):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                done = 0
                env_dict = env.reset()
                # agent.reset_randomness()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                    env_dict = env.reset()
                    # agent.reset_randomness()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]
                episode_reward = 0
                step = 0
                while not done:
                    # env.render()
                    step += 1
                    action = agent.choose_action(state, desired_goal)
                    next_env_dict, reward, done, info = env.step(action)

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())
                    episode_dict["next_state"].append(next_state.copy())
                    episode_dict["next_achieved_goal"].append(next_achieved_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()
                    episode_reward += reward
                # agent.store(dc(episode_dict))
                mb.append(dc(episode_dict))

            agent.store(mb)
            actor_loss, critic_loss = 0, 0
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
            agent.update_networks()

        # if MPI.COMM_WORLD.Get_rank() == 0:
        ram = psutil.virtual_memory()
        succes_rate, running_reward, episode_reward = eval_agent(env, agent)
        total_success_rate.append(succes_rate)
        print(f"Epoch:{epoch}| "
              f"Running_reward:{running_reward[-1]:.3f}| "
              f"EP_reward:{episode_reward:.3f}| "
              f"Memory_length:{len(agent.memory)}| "
              f"Duration:{time.time() - start_time:3.3f}| "
              f"Actor_Loss:{actor_loss:3.3f}| "
              f"Critic_Loss:{critic_loss:3.3f}| "
              f"Success rate:{succes_rate:.3f}| "
              f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
        agent.save_weights()

    # player = Play(env, agent)
    # player.evaluate()
    #
    plt.figure()
    # plt.subplot(311)
    plt.plot(np.arange(0, MAX_EPOCHS), total_success_rate)
    plt.title("Reward")
    #
    # plt.subplot(312)
    # plt.plot(np.arange(0, MAX_EPISODES), total_ac_loss)
    # plt.title("Actor loss")
    #
    # plt.subplot(313)
    # plt.plot(np.arange(0, MAX_EPISODES), total_cr_loss)
    # plt.title("Critic loss")
    #
    # plt.show()
