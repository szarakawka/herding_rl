import copy
import numpy as np


class Agent:

    def __init__(self, env):
        self.x = 0
        self.y = 0
        self.radius = env.agent_radius
        self.dog_list = None
        self.sheep_list = None

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def set_lists(self, dog_list, sheep_list):
        self.dog_list = dog_list
        self.sheep_list = sheep_list


class PassiveAgent(Agent):

    def move(self):
        raise NotImplementedError


class ActiveAgent(Agent):

    def __init__(self, env):
        super().__init__(env)
        if env.use_tan_to_center:
            self.observation = np.ndarray(shape=(2, env.ray_count + 1), dtype=float)
        else:
            self.observation = np.ndarray(shape=(2, env.ray_count), dtype=float)

    def move(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError
