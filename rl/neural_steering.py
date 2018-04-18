import sys
import os.path
import json
import numpy as np
from tensorforce.agents import Agent


class NeuralSteering:

    def __init__(self, env, save_dir, deterministic=True):

        self.save_dir = save_dir
        self.env = env
        self.deterministic = deterministic

        # training spec
        with open(os.path.join(save_dir, 'training_spec.json'), 'r') as f:
            training_spec = json.load(f)

        self.repeat_actions = training_spec["repeat_actions"]
        self.momentum = training_spec["momentum"]
        self.max_episode_timesteps = training_spec["max_episode_timesteps"]
        self.num_agent_clones = env.gym.dog_count

        # network spec
        with open(os.path.join(save_dir, 'network_spec.json'), 'r') as f:
            network_spec = json.load(f)

        if os.path.exists(os.path.join(save_dir, 'preprocessing_spec.json')):
            with open(os.path.join(save_dir, 'preprocessing_spec.json'), 'r') as f:
                preprocessing_spec = json.load(f)
        else:
            preprocessing_spec = None

        # agent spec
        with open(os.path.join(save_dir, 'agent_spec.json'), 'r') as f:
            agent_spec = json.load(f)

        self.agent = Agent.from_spec(
            spec=agent_spec,
            kwargs=dict(
                states=env.states,
                actions=env.actions,
                network=network_spec,
                states_preprocessing=preprocessing_spec
            )
        )

        self.load_model()

        # self.is_monitor = isinstance(env.gym, gym.wrappers.Monitor)
        # dog_count = env.gym.dog_count if not self.is_monitor else env.gym.env.dog_count
        # self.agent = MultiAgentWrapper(
        #         self.agent_type,
        #         dict(
        #             states=self.env.states,
        #             actions=self.env.actions,
        #             network=self.network_spec
        #         ),
        #         dog_count)

    def load_model(self):
        if os.path.isfile(os.path.join(self.save_dir, 'model', 'checkpoint')):
            self.agent.restore_model(os.path.join(self.save_dir, 'model/'))
            print('model loaded')
        else:
            print('model not loaded!')
        sys.stdout.flush()

    def show_simulation(self):
        while True:
            state = self.env.reset()
            self.agent.reset()
            timestep = 0
            while True:
                action = self.agent.act(states=state)
                terminal = False
                for _ in range(3):
                    state, terminal, reward = self.env.execute(actions=action)
                    timestep += 1
                    self.env.gym.render()
                    if terminal is True:
                        break
                if terminal is True or timestep >= 2000:
                    break

    def show_simulation_round_robin(self):
        while True:
            state = self.env.reset()
            self.agent.reset()
            default_action = np.zeros(shape=self.env.actions['shape'], dtype=np.float32)
            actions = [default_action for _ in range(self.num_agent_clones)]

            current_timestep = 0
            which_clone = 0

            terminal = False

            # time step (within episode) loop
            while True:

                # Round robin all agents acting loop
                which_clone += 1
                which_clone %= self.num_agent_clones

                actions[which_clone] = self.agent.act(states=state[which_clone], deterministic=self.deterministic)

                for repeat in range(self.repeat_actions):
                    state, terminal, _ = self.env.execute(actions=actions)
                    self.env.gym.render()
                    if terminal:
                        break

                if self.max_episode_timesteps is not None and current_timestep >= self.max_episode_timesteps:
                    terminal = True

                current_timestep += 1

                if terminal or self.agent.should_stop():  # TODO: should_stop also terminate?
                    break

                if not self.momentum:
                    actions[which_clone] = default_action
