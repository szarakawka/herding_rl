import sys
import os
import gym
from tensorforce.agents import TRPOAgent
from tensorforce.execution import Runner
from rl.multi_agent_wrapper import MultiAgentWrapper
import threading
from statistics import mean
EXIT = -1
NOOP = 0
SAVE = 1
flag = NOOP


class Learning:

    def __init__(
            self,
            env,
            agent_type=TRPOAgent,
            repeat_actions=5,
            max_episode_timesteps=8000
    ):
        self.save_dir = os.path.dirname(__file__) + '/model/'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.network_spec = [
            dict(type='dense', size=128),
            dict(type='dense', size=64)
        ]

        self.env = env

        self.agent_type = agent_type

        self.is_monitor = isinstance(env.gym, gym.wrappers.Monitor)

        dog_count = env.gym.dog_count if not self.is_monitor else env.gym.env.dog_count
        self.agent = MultiAgentWrapper(
                self.agent_type,
                dict(
                    states=self.env.states,
                    actions=self.env.actions,
                    network=self.network_spec
                ),
                dog_count)

        self.repeat_actions = repeat_actions
        self.max_episode_timesteps = max_episode_timesteps
        self.runner = Runner(agent=self.agent, environment=self.env, repeat_actions=repeat_actions)
        self.instance_episodes = 0
        self.terminal_reward = env.gym.max_episode_reward if not self.is_monitor else env.gym.env.max_episode_reward
        sys.stdout.flush()

    def _log_data(self, r, info):
        with open(self.save_dir + '/out.log', 'a+') as f:
            message = '{ep} {ts} {rw} {info}\n'.format(ep=r.episode, ts=r.timestep, rw=r.episode_rewards[-1], info=info)
            f.write(message)
            print(message)
        sys.stdout.flush()

    def episode_finished(self, r, _):
        global flag, EXIT, SAVE, NOOP
        save_frequency = 50
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
