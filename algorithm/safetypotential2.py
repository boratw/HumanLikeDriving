
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

import tensorflow.compat.v1 as tf
from lanetrace import LaneTrace

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

class SafetyPotential:
    def __init__(self):
        pass
    
    def predict_default(self, other_vcs):
        pred_prob = []
        pred_route = []
        for vcs in other_vcs:
            x = vcs[0]
            y = vcs[1]
            vx = vcs[4]
            vy = vcs[5]
            route = []
            for j in range(3):
                x += vx
                y += vy
                route.extend([x, y])

            pred_prob.append([1.])
            pred_route.append([route])
        return pred_prob, pred_route
            

    def get_potential(self, target_velocity_in_scenario, steer, agent_tr, agent_v, other_vcs):
        target_velocity = target_velocity_in_scenario / 3.6
        v_prob = [0.] * 12

        if target_velocity != 0.:
            pred_prob, pred_route = self.predict_default(other_vcs)

            potential = np.zeros((9, 16))


            nx = [3.0]
            ny = [0.0]
            theta = 0.0
            for i in range(8):
                nx.append(nx[-1] + 4.05 * np.cos(theta))
                ny.append(ny[-1] + 4.05 * np.sin(theta))
                theta += steer * -0.4

            for vcsi, vcs in enumerate(other_vcs):
                f = np.sqrt(vcs[4] ** 2 + vcs[5] ** 2)
                fx = vcs[0] + (f + 1.0) * 5.0
                fy = vcs[1] + (f + 1.0) * 5.0
                route = pred_route[vcsi]
                for i in range(len(pred_prob[vcsi])):
                    prob = pred_prob[vcsi][i]
                    for k in range(9):

                        dx = vcs[0] - nx[k]
                        dy = vcs[1] - ny[k]
                        d = (3. - np.sqrt(dx * dx + dy * dy))
                        if d > 1.:
                            d = 1.
                        if potential[k][0] < (d * prob):
                            potential[k][0] = d * prob

                        dx += fx
                        dy += fy
                        d = (3. - np.sqrt(dx * dx + dy * dy))
                        if d > 1.:
                            d = 1.
                        if potential[k][0] < (d * prob):
                            potential[k][0] = d * prob

                    for j in range(0, 6, 2):
                        for k in range(9):

                            dx = route[i][j] - nx[k]
                            dy = route[i][j+1] - ny[k]
                            d = (3. - np.sqrt(dx * dx + dy * dy))
                            if d > 1.:
                                d = 1.
                            if potential[k][j + 1] < (d * prob):
                                potential[k][j + 1] = d * prob

                            dx += fx
                            dy += fy
                            d = (3. - np.sqrt(dx * dx + dy * dy))
                            if d > 1.:
                                d = 1.
                            if potential[k][j + 1] < (d * prob):
                                potential[k][j + 1] = d * prob

            
            v_prob[0] = max([potential[0][0], potential[1][2], potential[2][4], potential[3][6]])
            v_prob[1] = max([potential[0][0], potential[1][1], potential[2][2], potential[3][3]])
            v_prob[2] = max([potential[0][0], potential[1][1] * 0.5 + potential[2][1] * 0.5, potential[3][2], potential[4][3] * 0.5 +  potential[5][3] * 0.5])
            v_prob[3] = max([potential[0][0], potential[1][0], potential[2][1], potential[4][2], potential[6][3]])
            v_prob[4] = max([potential[0][0], potential[1][0], potential[2][1] * 0.5 + potential[3][1] * 0.5,  potential[5][2], potential[7][3] * 0.5 +  potential[8][3] * 0.5])
            v_prob[5] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][1] + potential[6][2]])
            v_prob[6] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][1] * 0.5 + potential[4][1] * 0.5 + potential[7][2]])
            v_prob[7] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][1] + potential[8][2]])
            v_prob[8] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][1] * 0.5 + potential[5][1] * 0.5])
            v_prob[9] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][0], potential[5][1] ])
            v_prob[10] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][0], potential[5][1] * 0.5 + potential[6][1] * 0.5])
            v_prob[11] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][0], potential[5][0], potential[6][1]])
            
            return v_prob
