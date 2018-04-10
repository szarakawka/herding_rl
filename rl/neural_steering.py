from tensorforce.agents import TRPOAgent
from rl.multi_agent_wrapper import MultiAgentWrapper
from rl.env_wrapper import EnvWrapper, OpenAIGymTensorforceWrapper
import herding


network_spec = [
    dict(type='dense', size=128),
    dict(type='dense', size=64)
]

env = OpenAIGymTensorforceWrapper(
    EnvWrapper(
        dog_count=3,
        sheep_count=5,
        agent_layout=herding.constants.AgentLayout.LAYOUT1,
        use_tan_to_center=True
    )
)

agent_type = TRPOAgent
agent = MultiAgentWrapper(
    agent_type,
    dict(
        states=env.states,
        actions=env.actions,
        network=network_spec,
    ),
    env.gym.dog_count)

agent.load_model('./model')

while True:
    state = env.reset()
    agent.reset()
    timestep = 0
    while True:
        action = agent.act(states=state)
        terminal = False
        for _ in range(5):
            state, terminal, reward = env.execute(actions=action)
            timestep += 1
            env.gym.render()
        if terminal is True or timestep == 50000:
            timestep = 0
            break