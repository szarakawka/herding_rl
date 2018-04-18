import sys
import os
import gym
import json
import numpy as np
from shutil import copyfile
from tensorforce.agents import Agent
import herding
from rl.env_wrapper import OpenAIGymTensorforceWrapper
from rl.multi_clones_runner import MultiClonesRoundRobinRunner

import threading

EXIT = -1
NOOP = 0
SAVE = 1
SAVE_FREQUENCY = 50
flag = NOOP


class Learning:

    def __init__(
            self,
            save_dir,
            env_spec_filepath,
            agent_spec_filepath,
            network_spec_filepath=None,
            preprocessing_spec_filepath=None,
            training_spec_filepath=None,
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
            herding.Herding.from_spec(env_spec_filepath),
            monitor=monitor,
            monitor_safe=monitor_safe,
            monitor_video=monitor_video,
            visualize=visualize)

        copyfile(env_spec_filepath, os.path.join(self.save_dir, 'env_spec.json'))

        if training_spec_filepath is not None:
            with open(training_spec_filepath, 'r') as f:
                training_spec = json.load(f)
            copyfile(training_spec_filepath, os.path.join(save_dir, 'training_spec.json'))
        else:
            raise FileNotFoundError("no training spec file")

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

        # add tensorboard support
        agent_spec['summarizer'] = {
            "directory": self.save_dir,
            "labels": ["rewards"],
            "seconds": 10
        }

        copyfile(agent_spec_filepath, os.path.join(self.save_dir, 'agent_spec.json'))

        self.is_monitor = isinstance(self.env.gym, gym.wrappers.Monitor)
        dog_count = self.env.gym.dog_count if not self.is_monitor else self.env.gym.env.dog_count

        agent_additional_spec = dict(
            states=self.env.states,
            actions=self.env.actions,
            network=network_spec,
            states_preprocessing=preprocessing_config
        )

        # if dog_count == 1:
        self.agent = Agent.from_spec(spec=agent_spec, kwargs=agent_additional_spec)
        # else:
        # self.agent = MultiAgentWrapper(agent_spec=agent_spec,
        #                                agent_additional_parameters=agent_additional_spec,
        #                                agents_count=dog_count)

        self.max_episode_timesteps = training_spec['max_episode_timesteps']
        self.runner = MultiClonesRoundRobinRunner(
            agent=self.agent,
            num_agent_clones=dog_count,
            environment=self.env,
            momentum=training_spec['momentum'],
            repeat_actions=training_spec['repeat_actions'])
        self.instance_episodes = 0
        self.terminal_reward =\
            self.env.gym.max_episode_reward if not self.is_monitor else self.env.gym.env.max_episode_reward
        sys.stdout.flush()

    def _log_data(self, r, info):
        with open(self.save_dir + '/out.log', 'a+') as f:
            message = 'Ep. {ep} timestep={ts} last_R={rw:.2f} 10_perc_of_last_50_R={rw50:.2f} {info}\n'. \
                format(ep=r.episode,
                       ts=r.timestep,
                       rw=r.episode_rewards[-1],
                       rw50=np.percentile(r.episode_rewards[-50:], 10),
                       info=info)
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
        if len(r.episode_rewards) >= 50 and np.percentile(r.episode_rewards[-50:], 10) > self.terminal_reward:
            self.save_model()
            return False

        return True

    def learn(self):
        self.runner.run(
            episode_finished=self.episode_finished,
            max_episode_timesteps=self.max_episode_timesteps)

    def stop_learning(self):
        self.agent.stop = True

    def load_model(self):
        if os.path.isfile(os.path.join(self.save_dir, 'model', 'checkpoint')):
            self.agent.restore_model(os.path.join(self.save_dir, 'model/'))
            print('model loaded')
        else:
            print('model not loaded!')
        sys.stdout.flush()

    def save_model(self):
        self.agent.save_model(os.path.join(self.save_dir, 'model/'))
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
