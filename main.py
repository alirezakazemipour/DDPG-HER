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

ENV_NAME = "FetchPickAndPlace-v1"
INTRO = False
MAX_EPOCHS = 200
MAX_CYCLES = 50
num_updates = 40
MAX_EPISODES = 16 // os.cpu_count()
memory_size = 7e+5 // 50 // os.cpu_count()
batch_size = 128
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

if INTRO:
    print(f"state_shape:{state_shape}\n"
          f"number of actions:{n_actions}\n"
          f"action boundaries:{action_bounds}")
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

    global_running_r = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        for cycle in range(MAX_CYCLES):
            for episode in range(MAX_EPISODES):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "reward": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": [],
                    "next_desired_goal": []}
                done = 0
                env_dict = env.reset()
                agent.reset_randomness()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                    env_dict = env.reset()
                    agent.reset_randomness()
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]
                episode_reward = 0
                step = 0
                while not done:
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
                    episode_reward += reward
                agent.store(dc(episode_dict))

                if episode == 0:
                    global_running_r.append(episode_reward)
                else:
                    global_running_r.append(global_running_r[-1] * 0.99 + 0.01 * episode_reward)
            actor_loss, critic_loss = 0, 0
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
            agent.update_networks()

        if MPI.COMM_WORLD.rank == 0:
            ram = psutil.virtual_memory()
            print(f"Epoch:{epoch}| "
                  f"EP_running_r:{global_running_r[-1]:.3f}| "
                  f"EP_reward:{episode_reward:.3f}| "
                  f"Memory_length:{len(agent.memory)}| "
                  f"Duration:{time.time() - start_time:3.3f}| "
                  f"Actor_Loss:{actor_loss:3.3f}| "
                  f"Critic_Loss:{critic_loss:3.3f}| "
                  f"Success rate:{info['is_success']}| "
                  f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")

    # agent.save_weights()
    # player = Play(env, agent)
    # player.evaluate()
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(np.arange(0, MAX_EPISODES), global_running_r)
    # plt.title("Reward")
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
