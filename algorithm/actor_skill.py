
from collections import deque
import random

class ActorSkill():
    def __init__(self):
        self.z_a = 0.
        self.steer_deque = deque(maxlen=5)
        self.brk_buffer = 0.
        self.yaw_buffer = 0.
        self.steer_buffer = 0.
    

    def adjust_action(self, steer):
        steer += self.z_a * 0.333333
        return steer

    def inference(self, steer, yaw_diff, velocity, brk):
        self.steer_deque.append(steer)
        self.brk_buffer = self.brk_buffer * 0.95 + brk * 0.05
        self.yaw_buffer = self.yaw_buffer * 0.75 + yaw_diff * 0.25
        self.steer_buffer = self.steer_buffer * 0.8 + self.steer_deque[-1] * velocity * 0.02

        yaw_diff = self.steer_buffer * 1. - self.yaw_buffer * 0.25 + (self.brk_buffer * 3 + self.z_a) * 5
        if yaw_diff > 0.5:
            self.z_a -= 0.04
        elif yaw_diff > 0.1:
            self.z_a += (yaw_diff - 0.1) * -0.1
        elif yaw_diff < -0.1:
            self.z_a += (yaw_diff + 0.1) * -0.1
        elif yaw_diff < -0.5:
            self.z_a += 0.04
        self.z_a = self.z_a
