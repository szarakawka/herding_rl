import herding
from gym import spaces
from tensorforce.contrib.openai_gym import OpenAIGym


class OpenAIWrapper(OpenAIGym):

    def __init__(self, env, gym_id):
        super().__init__(gym_id)
        self.gym_id = gym_id
        self.gym = env


class EnvWrapper(herding.Herding):

    def step(self, action):
        state, reward, terminal, _ = super().step(action)
        newState = []
        for i, _ in enumerate(self.dog_list):
            s = state[i]
            s = s.flatten()
            newState.append(s)

        return newState, reward, terminal, _

    def reset(self):
        state = super().reset()
        newState = []
        for i, _ in enumerate(self.dog_list):
            s = state[i]
            s = s.flatten()
            newState.append(s)

        return newState

    @property
    def action_space(self):
        dim = 3 if self.rotation_mode is herding.constants.RotationMode.FREE else 2
        singleActionSpace = spaces.Box(-1, 1, (dim,))
        return singleActionSpace

    @property
    def observation_space(self):
        return spaces.Box(-1, 1, ((self.ray_count + 1) * 2,))
