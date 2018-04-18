from herding.envs.assets import constants
from herding.envs.assets.herding import Herding
# from herding.manual_steering import play

import gym

gym.envs.registration.register(
    id='herding-singleDog-v0',
    entry_point='herding.envs:HerdingSingleDog',
    timestep_limit=8000,
    nondeterministic=False
)
