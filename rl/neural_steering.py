import sys
import os
import json
from tensorforce.agents import Agent


class NeuralSteering:

    def __init__(self, env, save_dir):

        self.save_dir = save_dir
        self.env = env

        # network spec
        with open(os.path.join(save_dir, 'network_spec.json'), 'r') as f:
            network_spec = json.load(f)

        with open(os.path.join(save_dir, 'preprocessing_spec.json'), 'r') as f:
            preprocessing_spec = json.load(f)

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
        if os.path.isfile(os.path.join(self.save_dir, 'checkpoint')):
            self.agent.restore_model(self.save_dir)
            print('Model loaded.')
        else:
            print('Model not loaded!')
        sys.stdout.flush()

    def show_simulation(self):
        while True:
            state = self.env.reset()
            self.agent.reset()
            timestep = 0
            while True:
                action = self.agent.act(states=state)
                terminal = False
                for _ in range(5):
                    state, terminal, reward = self.env.execute(actions=action)
                    timestep += 1
                    self.env.gym.render()
                if terminal is True or timestep == 50000:
                    timestep = 0
                    break
