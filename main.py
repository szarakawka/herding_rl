import herding
import os.path
from rl.env_wrapper import EnvWrapper, OpenAIGymTensorforceWrapper
from rl.learning import Learning
from rl.neural_steering import NeuralSteering


class Mode:
    PLAY, TRAIN, TEST = range(3)


mode = Mode.TRAIN


save_dir = 'experiments_logs/003/'
env_spec_filepath = 'rl/current_env_spec.json'
agent_spec_filepath = 'rl/current_agent_spec.json'
network_spec_filepath = 'rl/current_network_spec.json'
preprocessing_spec_filepath = 'rl/current_preprocessing_spec.json'


if mode == Mode.PLAY:
    env = herding.Herding.from_spec(env_spec_filepath)
    herding.play(env)
elif mode == Mode.TRAIN:
    rl = Learning(save_dir=save_dir,
                  env_spec_filepath=env_spec_filepath,
                  agent_spec_filepath=agent_spec_filepath,
                  network_spec_filepath=network_spec_filepath,
                  preprocessing_spec_filepath=preprocessing_spec_filepath,
                  max_episode_timesteps=1000)
    # rl.load_model()
    rl.learn()
elif mode == Mode.TEST:
    env = herding.Herding.from_spec(os.path.join(save_dir, 'env_spec.json'))
    ns = NeuralSteering(
        env=OpenAIGymTensorforceWrapper(EnvWrapper.from_herding(env)),
        save_dir=save_dir)
    ns.show_simulation()
else:
    raise ValueError('bad mode')

