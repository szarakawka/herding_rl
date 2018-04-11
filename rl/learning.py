import sys
import os
import gym
import json
from shutil import copyfile
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import herding
from rl.env_wrapper import EnvWrapper, OpenAIGymTensorforceWrapper

# from rl.multi_agent_wrapper import MultiAgentWrapper
import threading
from statistics import mean
EXIT = -1
NOOP = 0
SAVE = 1
SAVE_FREQUENCY = 10
flag = NOOP


class Learning:

    def __init__(
            self,
            save_dir,
            env_spec_filepath,
            agent_spec_filepath,
            network_spec_filepath=None,
            preprocessing_spec_filepath=None,
            repeat_actions=3,       # was: repeat actions=5
            max_episode_timesteps=1000,      # max_episode_timesteps=8000
            monitor=None,
            monitor_safe=False,
            monitor_video=0,
            visualize=False
    ):

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # env spec
        self.env = OpenAIGymTensorforceWrapper(
            EnvWrapper.from_herding(herding.Herding.from_spec(env_spec_filepath)),
            monitor=monitor,
            monitor_safe=monitor_safe,
            monitor_video=monitor_video,
            visualize=visualize)

        copyfile(env_spec_filepath, os.path.join(self.save_dir, 'env_spec.json'))

        # network spec
        if network_spec_filepath is not None:
            with open(network_spec_filepath, 'r') as f:
                network_spec = json.load(f)
            copyfile(network_spec_filepath, os.path.join(self.save_dir, 'network_spec.json'))
        else:
            network_spec = [
                dict(type='dense', size=128),
                dict(type='dense', size=64)
            ]
            with open(os.path.join(self.save_dir, 'network_spec.json'), 'w') as f:
                json.dump(network_spec, f)

        preprocessing_config = None
        if preprocessing_spec_filepath is not None:
            with open(preprocessing_spec_filepath, 'r') as f:
                preprocessing_config = json.load(f)
            copyfile(preprocessing_spec_filepath, os.path.join(self.save_dir, 'preprocessing_spec.json'))

        # agent spec
        with open(agent_spec_filepath, 'r') as fp:
            agent_spec = json.load(fp=fp)
        copyfile(agent_spec_filepath, os.path.join(self.save_dir, 'agent_spec.json'))
        self.agent = Agent.from_spec(
            spec=agent_spec,
            kwargs=dict(
                states=self.env.states,
                actions=self.env.actions,
                network=network_spec,
                states_preprocessing=preprocessing_config
            )
        )

        self.is_monitor = isinstance(self.env.gym, gym.wrappers.Monitor)

        # dog_count = env.gym.dog_count if not self.is_monitor else env.gym.env.dog_count
        # self.agent = MultiAgentWrapper(
        #         self.agent_type,
        #         dict(
        #             states=self.env.states,
        #             actions=self.env.actions,
        #             network=self.network_spec
        #         ),
        #         dog_count)

        self.repeat_actions = repeat_actions
        self.max_episode_timesteps = max_episode_timesteps
        self.runner = Runner(agent=self.agent, environment=self.env, repeat_actions=repeat_actions)
        self.instance_episodes = 0
        self.terminal_reward = self.env.gym.max_episode_reward if not self.is_monitor else self.env.gym.env.max_episode_reward
        sys.stdout.flush()

    def _log_data(self, r, info):
        with open(self.save_dir + '/out.log', 'a+') as f:
            message = '{ep} {ts} {rw} {info}\n'.format(ep=r.episode, ts=r.timestep, rw=r.episode_rewards[-1], info=info)
            f.write(message)
            print(message)
        sys.stdout.flush()

    def episode_finished(self, r, _):
        global flag, EXIT, SAVE, NOOP
        save_frequency = SAVE_FREQUENCY
        info = ''
        self.instance_episodes += 1

        if self.instance_episodes >= save_frequency and r.episode % save_frequency == 0:
            self.save_model()

        self._log_data(r, info)

        if flag == SAVE:
            self.save_model()
            return False
        if flag == EXIT:
            return False
        if len(r.episode_rewards) >= 50 and mean(r.episode_rewards[-50:]) > self.terminal_reward:
            self.save_model()
            return False

        return True

    def learn(self):
        self.runner.run(episode_finished=self.episode_finished, max_episode_timesteps=self.max_episode_timesteps)

    def stop_learning(self):
        self.agent.stop = True

    def load_model(self):
        if os.path.isfile(self.save_dir + '/checkpoint'):
            self.agent.load_model(self.save_dir)
            print('model loaded')
            sys.stdout.flush()

    def save_model(self):
        self.agent.save_model(self.save_dir)
        print('model saved')
        sys.stdout.flush()


class InputThread(threading.Thread):

    def run(self):
        global flag, EXIT, NOOP, SAVE
        while flag is not EXIT:
            text = input()
            if text is 'q':
                flag = EXIT
            if text is 's':
                flag = SAVE
                break


class LearningThread(threading.Thread):

    def __init__(self, params):
        super().__init__()
        self.learning = Learning(**params)

    def run(self):
        self.learning.load_model()
        self.learning.learn()
