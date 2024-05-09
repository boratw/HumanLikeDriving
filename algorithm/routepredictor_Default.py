

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

class RoutePredictor_Default:
    def __init__(self):

        pass

    def Assign_NPCS(self, npcs):
        self.npcs = npcs
        self.agent_count = len(npcs)
    
    def Reset(self):
        pass

    def Get_Predict_Result(self, transforms, velocities,  npc_lights, impatiences):
        self.pred_prob = []
        self.pred_route = []
        for i in range(self.agent_count):
            x = transforms[i].location.x
            y = transforms[i].location.y
            vx = velocities[i].x
            vy = velocities[i].y
            route = []
            for j in range(3):
                x += vx
                y += vy
                route.extend([x, y])

            self.pred_prob.append([1.])
            self.pred_route.append([route])
            