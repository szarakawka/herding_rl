import herding
import gym
import numpy as np
from gym import spaces
from tensorforce import TensorForceError
from tensorforce.environments import Environment


# class copied from TensorForce, slightly modified to work
# without the need of gym_id parameter (registered envs only), instead directly passing custom env
class OpenAIGymTensorforceWrapper(Environment):

    def __init__(self, gym_env, monitor=None, monitor_safe=False, monitor_video=0, visualize=False):
        """
        Initialize OpenAI Gym.

        Args:
            gym_env: OpenAI Gym environment. See https://gym.openai.com/envs
            monitor: Output directory. Setting this to None disables monitoring.
            monitor_safe: Setting this to True prevents existing log files to be overwritten. Default False.
            monitor_video: Save a video every monitor_video steps. Setting this to 0 disables recording of videos.
            visualize: If set True, the program will visualize the trainings of gym's environment. Note that such
                visualization is probabily going to slow down the training.
        """

        # self.gym_id = gym_id
        # self.gym = gym.make(gym_id)  # Might raise gym.error.UnregisteredEnv or gym.error.DeprecatedEnv
        self.gym = gym_env
        self.visualize = visualize

        if monitor:
            if monitor_video == 0:
                video_callable = False
            else:
                video_callable = (lambda x: x % monitor_video == 0)
            self.gym = gym.wrappers.Monitor(self.gym, monitor, force=not monitor_safe, video_callable=video_callable)

    def __str__(self):
        return 'OpenAIGym({})'.format('env')
        # return 'OpenAIGym({})'.format(self.gym_id)

    def close(self):
        self.gym.close()
        self.gym = None

    def reset(self):
        if isinstance(self.gym, gym.wrappers.Monitor):
            self.gym.stats_recorder.done = True
        return self.gym.reset()

    def execute(self, actions):
        if self.visualize:
            self.gym.render()
        # if the actions is not unique, that is, if the actions is a dict
        if isinstance(actions, dict):
            actions = [actions['action{}'.format(n)] for n in range(len(actions))]
        state, reward, terminal, _ = self.gym.step(actions)
        return state, terminal, reward

    @property
    def states(self):
        return OpenAIGymTensorforceWrapper.state_from_space(space=self.gym.observation_space)

    @staticmethod
    def state_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(shape=(), type='int')
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(shape=space.n, type='int')
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return dict(shape=space.num_discrete_space, type='int')
        elif isinstance(space, gym.spaces.Box):
            return dict(shape=tuple(space.shape), type='float')
        elif isinstance(space, gym.spaces.Tuple):
            states = dict()
            n = 0
            for space in space.spaces:
                state = OpenAIGymTensorforceWrapper.state_from_space(space=space)
                if 'type' in state:
                    states['state{}'.format(n)] = state
                    n += 1
                else:
                    for state in state.values():
                        states['state{}'.format(n)] = state
                        n += 1
            return states
        else:
            raise TensorForceError('Unknown Gym space.')

    @property
    def actions(self):
        return OpenAIGymTensorforceWrapper.action_from_space(space=self.gym.action_space)

    @staticmethod
    def action_from_space(space):
        if isinstance(space, gym.spaces.Discrete):
            return dict(type='int', num_actions=space.n)
        elif isinstance(space, gym.spaces.MultiBinary):
            return dict(type='bool', shape=space.n)
        elif isinstance(space, gym.spaces.MultiDiscrete):
            num_discrete_space = len(space.nvec)
            if (space.nvec == space.nvec[0]).all():
                return dict(type='int', num_actions=space.nvec[0], shape=num_discrete_space)
            else:
                actions = dict()
                for n in range(num_discrete_space):
                    actions['action{}'.format(n)] = dict(type='int', num_actions=space.nvec[n])
                return actions
        elif isinstance(space, gym.spaces.Box):
            if (space.low == space.low[0]).all() and (space.high == space.high[0]).all():
                return dict(type='float', shape=space.low.shape,
                            min_value=np.float32(space.low[0]),
                            max_value=np.float32(space.high[0]))
            else:
                actions = dict()
                low = space.low.flatten()
                high = space.high.flatten()
                for n in range(low.shape[0]):
                    actions['action{}'.format(n)] = dict(type='float', min_value=low[n], max_value=high[n])
                return actions
        elif isinstance(space, gym.spaces.Tuple):
            actions = dict()
            n = 0
            for space in space.spaces:
                action = OpenAIGymTensorforceWrapper.action_from_space(space=space)
                if 'type' in action:
                    actions['action{}'.format(n)] = action
                    n += 1
                else:
                    for action in action.values():
                        actions['action{}'.format(n)] = action
                        n += 1
            return actions
        else:
            raise TensorForceError('Unknown Gym space.')


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
        singleActionSpace = spaces.Box(-1, 1, (dim,), dtype=np.float32)
        return singleActionSpace

    @property
    def observation_space(self):
        return spaces.Box(-1, 1, ((self.ray_count + 1) * 2,), dtype=np.float32)
