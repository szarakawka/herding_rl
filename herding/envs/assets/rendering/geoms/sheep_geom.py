from .geom import *
from gym.envs.classic_control import rendering


class SheepGeom(Geom):
    BODY = 0

    def __init__(self, sheepObject):
        self.object = sheepObject

        self.body = Part(rendering.make_circle(self.object.radius, res=50))
        self.body.set_color(181 / 255, 185 / 255, 215 / 255)

    def get_parts(self):
        return [self.body.body]

    def update(self):
        self.body.set_pos(self.object.x, self.object.y)
