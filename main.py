import herding
from rl.env_wrapper import EnvWrapper, OpenAIGymTensorforceWrapper
from rl.learning import *
import atexit


rl = Learning(
    env=OpenAIGymTensorforceWrapper(
        EnvWrapper(
            dog_count=1,
            sheep_count=5,
            agent_layout=herding.constants.AgentLayout.LAYOUT1,
            use_tan_to_center=True
        )
    )
)


def exit_handler():
    global rl
    rl.save_model()


atexit.register(exit_handler)


rl.load_model()
rl.learn()

# env = herding.Herding(
#     sheep_count=10,
#     max_movement_speed=10,
#     rotation_mode=herding.constants.RotationMode.LOCKED_ON_HERD_CENTRE,
#     agent_layout=herding.constants.AgentLayout.LAYOUT1
# )
#
# herding.play(env)
