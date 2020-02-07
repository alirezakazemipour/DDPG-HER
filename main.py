import gym
from agent import Agent

ENV_NAME = "MountainCarContinuous-v0"
INTRO = False
MAX_EPISODES = 200

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.shape
n_actions = test_env.action_space.shape[0]
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
            _, r, done, _ = test_env.step(action)
            test_env.render()
else:
    env = gym.make(ENV_NAME)
    agent = Agent(n_states=state_shape, action_bounds=action_bounds)
    total_reward = 0
    for episode in range(MAX_EPISODES):
        agent.reset_randomness()
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, reward, done, next_state)
            episode_reward += reward
            if done:
                break
            state = next_state
        total_reward = episode_reward

