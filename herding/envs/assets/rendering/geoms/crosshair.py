from .geom import *
from gym.envs.classic_control import rendering


class Crosshair(Geom):

    def __init__(self, env):
        self.herd_centre_point = env.herd_centre_point
        crosshair_size = 10
        color = (0, 0, 0)
        self.vertical_bar = Part(rendering.Line((-crosshair_size - 1, 0), (crosshair_size, 0)))
        self.horizontal_bar = Part(rendering.Line((0, -crosshair_size - 1), (0, crosshair_size)))
        self.herd_circle = Part(rendering.make_circle(env.herd_target_radius, res=50, filled=False))

        self.vertical_bar.set_color(*color)
        self.horizontal_bar.set_color(*color)
        self.herd_circle.set_color(*color)

    def get_parts(self):
        return [self.vertical_bar.body, self.horizontal_bar.body, self.herd_circle.body]

    def update(self):
        self.horizontal_bar.set_pos(self.herd_centre_point[0], self.herd_centre_point[1])
        self.vertical_bar.set_pos(self.herd_centre_point[0], self.herd_centre_point[1])
        self.herd_circle.set_pos(self.herd_centre_point[0], self.herd_centre_point[1])
