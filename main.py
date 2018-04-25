from herding.envs.assets.herding import Herding
import os.path
import json
from rl.env_wrapper import OpenAIGymTensorforceWrapper
from rl.learning import Learning
from rl.neural_steering import NeuralSteering


class Mode:
    PLAY, TRAIN, TEST, RETRAIN = range(4)


mode = Mode.TRAIN


save_dir = 'experiments_logs/052_potential/'
env_spec_filepath = 'rl/current_env_spec.json'
training_spec_filepath = 'rl/current_training_spec.json'
agent_spec_filepath = 'rl/current_agent_spec.json'
network_spec_filepath = 'rl/current_network_spec.json'
preprocessing_spec_filepath = None      # 'rl/current_preprocessing_spec.json'


if mode == Mode.PLAY:
    env = Herding.from_spec(env_spec_filepath)
    from herding.manual_steering import play
    play(env)
elif mode == Mode.TRAIN or mode == Mode.RETRAIN:
    rl = Learning(save_dir=save_dir,
                  env_spec_filepath=env_spec_filepath,
                  agent_spec_filepath=agent_spec_filepath,
                  network_spec_filepath=network_spec_filepath,
                  preprocessing_spec_filepath=preprocessing_spec_filepath,
                  training_spec_filepath=training_spec_filepath,
                  # monitor=save_dir,
                  # monitor_video=10,
                  visualize=False)

    rl = Learning(save_dir=save_dir,
                  env_spec_filepath=env_spec_filepath,
                  agent_spec_filepath=agent_spec_filepath,
                  network_spec_filepath=network_spec_filepath,
                  preprocessing_spec_filepath=preprocessing_spec_filepath,
                  training_spec_filepath=training_spec_filepath,
                  # monitor=save_dir,
                  # monitor_video=10,
                  visualize=False)

    if mode == Mode.RETRAIN:
        rl.load_model()
    rl.learn()

elif mode == Mode.TEST:
    env = Herding.from_spec(os.path.join(save_dir, 'env_spec.json'))
    with open(os.path.join(save_dir, 'training_spec.json'), 'r') as f:
        training_spec = json.load(f)
    ns = NeuralSteering(
        env=OpenAIGymTensorforceWrapper(env),
        save_dir=save_dir,
        deterministic=False
        )
    ns.show_simulation_round_robin()

else:
    raise ValueError('bad mode')
