import copy
import numpy as np
from ..constants import AgentObservationAids, AgentObservationCompression


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
        self.observation = np.ndarray(shape=self.get_observation_array_shape(env), dtype=np.float32)

    @staticmethod
    def get_observation_array_shape(env):
        n_channels = 2 if env.agent_observations_compression == AgentObservationCompression.TWO_CHANNEL else 1

        size = env.ray_count
        if env.agent_observations_compression == AgentObservationCompression.TWO_CHANNEL_FLATTENED:
            size *= 2
        if env.agent_observations_aids is not AgentObservationAids.NO:
            if n_channels == 1:
                size += 2
            else:
                size += 1

        return n_channels, size

    def move(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError
