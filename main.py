import herding
import gym
from rl.env_wrapper import EnvWrapper
from rl.learning import *
import atexit

#rl = None


def exit_handler():
    global rl
    rl.save_model()


atexit.register(exit_handler)


rl = Learning(
    env=OpenAIWrapper(
        EnvWrapper(
            dog_count=1,
            sheep_count=5,
            agent_layout=herding.constants.AgentLayout.LAYOUT1
        ),
        'herding-env_v0')
)
rl.load_model()
rl.learn()
