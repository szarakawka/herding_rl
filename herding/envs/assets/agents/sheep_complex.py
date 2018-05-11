from .agent import PassiveAgent
from .sheep_simple import SheepSimple
import random


class SheepComplex(PassiveAgent):

    def __init__(self, env):
        super().__init__(env)
        self.max_movement_speed = env.max_movement_speed

    def move(self):
        result = random.randrange(0, 1000)
        if result < 5:
            print(result)
            direction = random.randrange(0, 3)
            if direction == 1:
                self.x = self.x + self.max_movement_speed
            elif direction == 2:
                self.x = self.x - self.max_movement_speed
            elif direction == 3:
                self.y = self.y + self.max_movement_speed
            elif direction == 0:
                self.y = self.y - self.max_movement_speed
        else:
            SheepSimple.move(self)
