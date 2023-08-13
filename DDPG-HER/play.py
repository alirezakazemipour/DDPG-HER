import torch
from torch import device
import numpy as np
import cv2
from gym import wrappers
# from mujoco_py import GlfwContext

# GlfwContext(offscreen=True)

# from mujoco_py.generated import const


class Play:
    def __init__(self, env, agent,ENV_NAME, max_episode=4):
        self.env = env
        # self.env = wrappers.Monitor(env, "./videos", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights(ENV_NAME)
        self.agent.set_to_eval_mode()
        self.device = "cpu" # device("mps" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def evaluate(self):

        for _ in range(self.max_episode):
            env_dict = self.env.reset()[0]
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]
            while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
                env_dict = self.env.reset()[0]
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, desired_goal, train_mode=False)
                next_env_dict, r, terminate, done, _ = self.env.step(action)
                next_state = next_env_dict["observation"]
                next_desired_goal = next_env_dict["desired_goal"]
                episode_reward += r
                state = next_state.copy()
                desired_goal = next_desired_goal.copy()
                # I = self.env.render(mode="human")  # mode = "rgb_array
                # self.env.viewer.cam.type = const.CAMERA_FREE
                # self.env.viewer.cam.fixedcamid = 0
                # I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                # cv2.imshow("I", I)
                # cv2.waitKey(2)
            print(f"episode_reward:{episode_reward:3.3f}")

        self.env.close()
