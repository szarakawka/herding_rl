import herding
from rl.env_wrapper import EnvWrapper, OpenAIGymTensorforceWrapper
from rl.learning import *


env = herding.Herding(
    dog_count=3,
    sheep_count=5,
    max_movement_speed=10,
    rotation_mode=herding.constants.RotationMode.FREE,
    agent_layout=herding.constants.AgentLayout.RANDOM,
    use_tan_to_center=True
)


# rl = Learning(env=OpenAIGymTensorforceWrapper(EnvWrapper(env)))
# rl.load_model()
# rl.learn()


herding.play(env)
