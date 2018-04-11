import herding
from rl.env_wrapper import EnvWrapper, OpenAIGymTensorforceWrapper
from rl.learning import *


env = herding.Herding(
    dog_count=1,
    sheep_count=5,
    max_movement_speed=10,
    rotation_mode=herding.constants.RotationMode.LOCKED_ON_HERD_CENTRE,
    agent_layout=herding.constants.AgentLayout.RANDOM,
    agent_observations_aids=herding.constants.AgentObservationAids.TO_MASS_CENTER,
    agent_observations_compression=herding.constants.AgentObservationCompression.TWO_CHANNEL
)

herding.play(env)

# save_dir = 'experiments_logs/001'
#
# rl = Learning(env=OpenAIGymTensorforceWrapper(
#                     EnvWrapper.from_herding(env),
#                     visualize=False),
#               save_dir=save_dir,
#               max_episode_timesteps=1000)
# # rl.load_model()
# rl.learn()


