import gym
import numpy as np
import random
import json
from gym import spaces
from . import constants
from . import agents
import math
import cmath


class Herding(gym.Env):

    metadata = {
        'render.modes': ['human']
    }

    def __init__(
            self,
            dog_count=1,
            sheep_count=3,
            agent_layout=constants.AgentLayout.RANDOM,
            sheep_type=constants.SheepType.SIMPLE,
            max_movement_speed=5,
            max_rotation_speed=90,
            continuous_sheep_spread_rate=1,
            ray_count=180,
            ray_length=600,
            field_of_view=180,
            rotation_mode=constants.RotationMode.FREE,
            agent_observations_representation=constants.AgentObservationRepresentation.TWO_CHANNEL_FLATTENED,
            agent_observations_aids=constants.AgentObservationAids.TO_MASS_CENTER,
            reward_type=constants.RewardCalculatorType.SCATTER_DIFFERENCE
    ):
        self.dog_count = dog_count
        self.sheep_count = sheep_count
        self.agent_layout = agent_layout
        self.sheep_type = sheep_type
        self.max_movement_speed = max_movement_speed
        self.max_rotation_speed = max_rotation_speed
        self.ray_count = ray_count
        self.ray_length = ray_length
        self.field_of_view = field_of_view
        self.rotation_mode = rotation_mode
        self.continuous_sheep_spread_rate = continuous_sheep_spread_rate
        self.agent_observations_representation = agent_observations_representation
        self.agent_observations_aids = agent_observations_aids
        self.reward_type = reward_type

        self.map_width = 800
        self.map_height = 600
        # self.map_width = 1280
        # self.map_height = 900
        self.agent_radius = 10

        self.herd_target_radius = 100
        self.max_episode_reward = 1000

        self.herd_centre_point = [0, 0]

        # helper arrays
        self.sheep_distances_to_herd_center = np.zeros(self.sheep_count, dtype=np.float32)
        self.sheep_in_target = np.zeros(self.sheep_count, dtype=np.float32)

        self.dog_list = None
        self.sheep_list = None
        self.dog_list, self.sheep_list = self._create_agents()
        self._set_agents_lists()

        self.reward_calculator = reward_type_factory(self.reward_type)(self)
        self.viewer = None
        self.agent_layout_function = AgentLayoutFunction.get_function(self.agent_layout)

    def step(self, action):

        for i, dog in enumerate(self.dog_list):
            dog.move(action[i])

        for sheep in self.sheep_list:
            sheep.move()

        self._update_herd_centre_point_and_helper_arrays()

        state = self._get_state()

        self.reward_calculator.do_step()

        reward = self.reward_calculator.reward
        done = self.is_done()

        return state, reward, done, {
            # "scatter": self.reward_calculator.scatter
        }

    def reset(self):
        self._set_up_agents()
        self._update_herd_centre_point_and_helper_arrays()
        self.reward_calculator.reset(self)
        return self._get_state()

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from .rendering.renderer import Renderer
            self.viewer = Renderer(self)

        self.viewer.render()

    def seed(self, seed=None):
        pass

    def close(self):
        self.viewer.close()

    @property
    def single_action_space(self):
        dim = 3 if self.rotation_mode is constants.RotationMode.FREE else 2
        single_action_space = spaces.Box(-1, 1, shape=(dim,), dtype=np.float32)
        return single_action_space

    # @property
    # def action_space(self):
    #     action_space = spaces.Tuple((self.single_action_space,) * self.dog_count)
    #     return action_space

    @property
    def action_space(self):
        return self.single_action_space

    @property
    def single_observation_space(self):
        shape = agents.Dog.get_observation_array_shape(self)
        single_observation_space = spaces.Box(-1, 1, shape=shape, dtype=np.float32)
        return single_observation_space

    # @property
    # def observation_space(self):
    #     observation_space = spaces.Tuple((self.single_observation_space,) * self.dog_count)
    #     return observation_space

    @property
    def observation_space(self):
        return self.single_observation_space

    def _create_agents(self):
        dog_list = []
        sheep_list = []
        sheep = agents.get_sheep_class(self.sheep_type)

        for i in range(self.dog_count):
            dog_list.append(agents.Dog(self))
        for i in range(self.sheep_count):
            sheep_list.append(sheep(self))

        return dog_list, sheep_list

    def _set_agents_lists(self):
        for agent in self.dog_list + self.sheep_list:
            agent.set_lists(self.dog_list, self.sheep_list)

    def _get_state(self):
        states = []
        for dog in self.dog_list:
            states.append(dog.get_observation())
        return states

    def _update_herd_centre_point_and_helper_arrays(self):
        self.herd_centre_point[0] = self.herd_centre_point[1] = 0
        for sheep in self.sheep_list:
            self.herd_centre_point[0] += sheep.x
            self.herd_centre_point[1] += sheep.y

        self.herd_centre_point[0] /= self.sheep_count
        self.herd_centre_point[1] /= self.sheep_count

        # update helper arrays
        self._calc_sheep_distances_to_herd_center()
        self._determine_if_sheep_inside_target_circle()

    def is_done(self):
        return bool(np.all(self.sheep_in_target))

    def _calc_sheep_distances_to_herd_center(self):
        for i, sheep in enumerate(self.sheep_list):
            self.sheep_distances_to_herd_center[i] = self._get_distance_to_herd_center(sheep)

    def _get_distance_to_herd_center(self, sheep):
        return math.sqrt(pow(sheep.x - self.herd_centre_point[0], 2) +
                         pow(sheep.y - self.herd_centre_point[1], 2))

    def _determine_if_sheep_inside_target_circle(self):
        for i in range(self.sheep_count):
            self.sheep_in_target[i] = \
                self.sheep_distances_to_herd_center[i] < self.herd_target_radius - self.agent_radius

    def _set_up_agents(self):
        self.agent_layout_function(self)

    @classmethod
    def from_spec(cls, spec_file_path):
        with open(spec_file_path, 'r') as f:
            spec = json.load(f, object_hook=constants.as_enum)
            return cls(**spec)


def reward_type_factory(reward_type):
    return {
        constants.RewardCalculatorType.SCATTER_DIFFERENCE: RelativeScatterRewardCalculator,
        constants.RewardCalculatorType.IN_TARGET_DIFFERENCE: InTargetDifferenceCircleRewardCalculator,
        constants.RewardCalculatorType.COMPLEX: ComplexRewardCalculator,
        constants.RewardCalculatorType.SCATTER_DIFFERENCE_SIGN: ScatterDifferenceSignRewardCalculator,
        constants.RewardCalculatorType.SCATTER_POTENTIAL_BASED: ScatterOnlyPotentialShapingRewardCalculator
    }[reward_type]


class RewardCalculator:

    def __init__(self, env):
        self.env = env
        self.reward = 0.

    def do_step(self):
        """This has to be overwritten by subclass."""
        raise NotImplementedError('Overwrite this method.')

    def reset(self, env):
        self.env = env
        self.reward = 0.


class ScatterBasedRewardCalculator(RewardCalculator):

    def __init__(self, env):
        super(ScatterBasedRewardCalculator, self).__init__(env)

        self.scatter = 0.
        self._calc_scatter()

    def reset(self, env):
        super(ScatterBasedRewardCalculator, self).reset(env)
        self._calc_scatter()

    def do_step(self):
        raise NotImplementedError('Overwrite this method.')

    def _calc_scatter(self):
        self.scatter = np.mean(self.env.sheep_distances_to_herd_center)


class RelativeScatterRewardCalculator(ScatterBasedRewardCalculator):

    def __init__(self, env):
        super(RelativeScatterRewardCalculator, self).__init__(env)
        self.first_scatter = self.scatter
        self.previous_scatter = self.scatter

    def do_step(self):
        self.previous_scatter = self.scatter
        self._calc_scatter()
        self.reward = (self.previous_scatter - self.scatter) * self.env.max_episode_reward / self.first_scatter

    def reset(self, env):
        super(RelativeScatterRewardCalculator, self).reset(env)
        self.first_scatter = self.scatter
        self.previous_scatter = self.scatter


class PotentialBasedShapingRewardCalculatorBase(RewardCalculator):

    def __init__(self, env):
        super(PotentialBasedShapingRewardCalculatorBase, self).__init__(env)
        self.potential = self.calc_state_potential()
        self.prev_potential = self.potential

    def calc_state_potential(self):
        raise NotImplementedError('Implement this: return potential')

    def do_step(self):
        self.prev_potential = self.potential
        self.potential = self.calc_state_potential()
        self.reward = self.potential - self.prev_potential

    def reset(self, env):
        super(PotentialBasedShapingRewardCalculatorBase, self).reset(env)
        self.potential = self.calc_state_potential()
        self.prev_potential = self.potential


class ScatterOnlyPotentialShapingRewardCalculator(PotentialBasedShapingRewardCalculatorBase):
    """
    Assumption: state potential is linearly dependent on sheep scatter only: psi = a * scatter + b
    Because: psi_target = R_max, and psi_initial = 0, then:
    a = - R_max / (initial_scatter - target_scatter)
    b = R_max * initial_scatter / (initial_scatter - target_scatter) : but b_coeffs cancels out
    when potentials are subtracted, so can be removed
    """

    def __init__(self, env):
        self.a_coefficient = 0.
        super(ScatterOnlyPotentialShapingRewardCalculator, self).__init__(env)
        self._calc_linear_coefficients()
        self.potential = self.calc_state_potential()
        self.prev_potential = self.potential

    def calc_state_potential(self):
        scatter = self._calc_scatter()
        return self.a_coefficient * scatter

    def reset(self, env):
        super(ScatterOnlyPotentialShapingRewardCalculator, self).reset(env)
        self._calc_linear_coefficients()
        self.potential = self.calc_state_potential()
        self.prev_potential = self.potential

    def _calc_linear_coefficients(self):
        initial_scatter = self._calc_scatter()
        target_scatter = self.env.herd_target_radius
        self.a_coefficient = - self.env.max_episode_reward / (initial_scatter - target_scatter)

    def _calc_scatter(self):
        return np.mean(self.env.sheep_distances_to_herd_center)


class ScatterDifferenceSignRewardCalculator(ScatterBasedRewardCalculator):

    def __init__(self, env):
        super(ScatterDifferenceSignRewardCalculator, self).__init__(env)
        self.previous_scatter = self.scatter

    def do_step(self):
        self.previous_scatter = self.scatter
        self._calc_scatter()
        self.reward = 1. if self.previous_scatter - self.scatter > 0. else -1.
        if self.previous_scatter == self.scatter:
            self.reward = 0.
        if self.env.is_done():
            self.reward += self.env.max_episode_reward

    def reset(self, env):
        super(ScatterDifferenceSignRewardCalculator, self).reset(env)
        self.previous_scatter = self.scatter


class InTargetDifferenceCircleRewardCalculator(RewardCalculator):

    def __init__(self, env):
        super(InTargetDifferenceCircleRewardCalculator, self).__init__(env)
        self.prev_sheep_in_target = np.zeros_like(env.sheep_in_target)
        np.copyto(self.prev_sheep_in_target, self.env.sheep_in_target)

    def do_step(self):
        self.reward = np.mean(self.env.sheep_in_target - self.prev_sheep_in_target) *\
                      self.env.max_episode_reward
        np.copyto(self.prev_sheep_in_target, self.env.sheep_in_target)

    def reset(self, env):
        super(InTargetDifferenceCircleRewardCalculator, self).reset(env)
        np.copyto(self.prev_sheep_in_target, self.env.sheep_in_target)


class ComplexRewardCalculator(RelativeScatterRewardCalculator):
    """
    Takes into account:
    1) scatter difference,
    2) time penalty,
    3) bonus for all sheep in herd center
    """

    def __init__(self, env):
        super(ComplexRewardCalculator, self).__init__(env)
        self.TIME_PENALTY = - self.env.max_episode_reward / 10000.

    def do_step(self):
        super(ComplexRewardCalculator, self).do_step()
        bonus = float(self.env.is_done()) * self.env.max_episode_reward
        self.reward += self.TIME_PENALTY + bonus

    def reset(self, env):
        super(ComplexRewardCalculator, self).reset(env)


class AgentLayoutFunction:

    @staticmethod
    def get_function(agent_layout):
        return{
            constants.AgentLayout.RANDOM: AgentLayoutFunction._random,
            constants.AgentLayout.EASY: AgentLayoutFunction._easy,
            constants.AgentLayout.LAYOUT1: AgentLayoutFunction._layout1,
            constants.AgentLayout.LAYOUT2: AgentLayoutFunction._layout2
        }[agent_layout]

    @staticmethod
    def _random(env):
        padding = 50
        for agent in env.dog_list + env.sheep_list:
            x = random.randint(agent.radius + padding, env.map_width - agent.radius - padding)
            y = random.randint(agent.radius + padding, env.map_height - agent.radius - padding)
            agent.set_pos(x, y)

    @staticmethod
    def _layout1(env):
        sheep_padding = 50
        for agent in env.sheep_list:
            x = random.randint(agent.radius + sheep_padding, env.map_width - agent.radius - sheep_padding)
            y = random.randint(agent.radius + sheep_padding + 200, env.map_height - agent.radius - sheep_padding)
            agent.set_pos(x, y)

        for i, agent in enumerate(env.dog_list):
            x = (i + 1) * (env.map_width / (env.dog_count + 1))
            y = 20
            agent.set_pos(x, y)


    @staticmethod
    def _easy(env):
        map_center = complex(env.map_width / 2., env.map_height / 2.)

        n_agents = len(env.sheep_list)
        phase_shift = random.random() * 2 * cmath.pi
        r = 2 * env.herd_target_radius
        for n, agent in enumerate(env.sheep_list):
            phase = phase_shift + 2 * cmath.pi * n / n_agents
            z = map_center + cmath.rect(r, phase)
            agent.set_pos(z.real, z.imag)

        n_agents = len(env.dog_list)
        phase_shift = random.random() * 2 * cmath.pi
        r = 4 * env.herd_target_radius
        for n, agent in enumerate(env.dog_list):
            phase = phase_shift + 2 * cmath.pi * n / n_agents
            z = map_center + cmath.rect(r, phase)
            agent.set_pos(z.real, z.imag)


    @staticmethod
    def _layout2(env):
        # TODO
        pass
