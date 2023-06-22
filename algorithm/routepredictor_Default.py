

import glob
import os
import sys
try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import cv2
import numpy as np
import datetime

class RoutePredictor:
    def __init__(self, agent_count=100):

        self.output_route_len = 5
        self.output_route_num = 1

    def Assign_NPCS(self, npcs):
        self.npcs = npcs
        self.agent_count = len(npcs)

    def Get_Predict_Result(self, close_npcs, npc_transforms, npc_velocities, actor_transform, actor_velocitiy):
        self.pred_prob = []
        self.pred_route = []
        for i in close_npcs:
            x = npc_transforms[i].location.x
            y = npc_transforms[i].location.y
            vx = npc_velocities[i].x * 0.75
            vy = npc_velocities[i].y * 0.75
            route = []
            for j in range(5):
                x += vx
                y += vy
                route.append([x, y])

            self.pred_prob.append(1.)
            self.pred_route.append(route)