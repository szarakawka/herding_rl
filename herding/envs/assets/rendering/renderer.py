from gym.envs.classic_control import rendering
from .geoms import *


class Renderer:

    def __init__(self, env):
        self.map_width = env.map_width
        self.map_height = env.map_height
        self.dog_list = env.dog_list
        self.sheep_list = env.sheep_list
        self.geom_list = self._initRenderObjects(env)
        self.viewer = rendering.Viewer(self.map_width, self.map_height)

        for geom in self.geom_list:
            self.viewer.geoms.extend(geom.get_parts())

    def _initRenderObjects(self, env):
        geom_list = []

        for dog in env.dog_list:
            geom_list.append(dog_geom.DogGeom(dog))

        for sheep in env.sheep_list:
            geom_list.append(sheep_geom.SheepGeom(sheep))

        geom_list.append(crosshair.Crosshair(env))

        return geom_list

    def render(self):
        for geom in self.geom_list:
            geom.update()

        self.viewer.render()

    def close(self):
        self.viewer.close()
