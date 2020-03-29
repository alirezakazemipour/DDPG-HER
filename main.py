import gym
from agent import Agent
import matplotlib.pyplot as plt
import numpy as np
from play import Play
import mujoco_py
import random

ENV_NAME = "FetchPickAndPlace-v1"
INTRO = False
MAX_EPOCHS = 200
MAX_CYCLES = 50
num_updates = 40
MAX_EPISODES = 16
memory_size = 1e+6
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
                  tau=tau)

    for epoch in range(MAX_EPOCHS):
        for cycle in range(MAX_CYCLES):
            # mini_batches = []
            for episode in range(MAX_EPISODES):

                done = 0
                step = 0
                env_dict = env.reset()
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                ep_s = []
                ep_a = []
                ep_r = []
                ep_d = []
                ep_ag = []
                ep_dg = []
                ep_next_s = []
                while not done:
                    action = agent.choose_action(state, desired_goal)
                    next_env_dict, reward, done, _ = env.step(action)
                    ep_s.append(state)
                    ep_a.append(action)
                    ep_r.append(reward)
                    ep_d.append(done)
                    ep_ag.append(next_env_dict["achieved_goal"])
                    ep_dg.append(desired_goal)
                    ep_next_s.append(next_env_dict["observation"])
                    agent.store((ep_s, ep_a, ep_r, ep_d, ep_ag, ep_dg, ep_next_s))
                # mini_batches.append((ep_s, ep_a, ep_r, ep_d, ep_ag, ep_dg, ep_next_s))
            # agent.store(mini_batches)
            for n_update in range(num_updates):
                agent.train()

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
