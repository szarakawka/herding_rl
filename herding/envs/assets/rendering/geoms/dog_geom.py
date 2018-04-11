from .geom import *
from gym.envs.classic_control import rendering
import math


class DogGeom(Geom):

    COLOR = {
        -1: (1, 0, 0),
        0: (0.5, 0.5, 0.5),
        1: (0, 1, 0)
    }

    def __init__(self, dogObject):
        self.object = dogObject

        self.body = Part(rendering.make_circle(self.object.radius, res=50))
        self.body.set_color(185 / 255, 14 / 255, 37 / 255)
        self.rays = []
        for _ in range(self.object.ray_count):
            self.rays.append(Part(rendering.Line((0, 0), (self.object.ray_length, 0))))

    def get_parts(self):
        parts = [self.body.body]
        for ray in self.rays:
            parts.append(ray.body)
        return parts

    def update(self):
        self.body.set_pos(self.object.x, self.object.y)
        for i, ray in enumerate(self.rays):
            ray.set_scale(1 - self.object.rays[0][i], 0)
            color = tuple(min(x * (1.5 - self.object.rays[0][i]), 1) for x in self.COLOR[self.object.rays[1][i]])
            ray.set_color(*color)
            rot = self.object.rotation - self.object.ray_radian[i]
            ray.set_rotation(rot)
            x = math.cos(rot) * self.object.radius
            y = math.sin(rot) * self.object.radius
            ray.set_pos(self.object.x + x, self.object.y + y)
