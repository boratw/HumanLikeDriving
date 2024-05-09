
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


def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

    
class SafetyPotential:
    def __init__(self, log_file_name = None, npc_count = 100):
        self.player = None
        self.npc_count = npc_count
        if log_file_name != None:
            self.log_file = open(log_file_name, "wt")
        else:
            self.log_file = None

    def Assign_Player(self, player):
        self.player = player
        self.fx_buffer = 0.
        self.fy_buffer = 0.

    def Change_Log_File(self, log_file_name):
        if self.log_file != None:
            self.log_file.close()
        self.log_file = open(log_file_name, "wt")



    def get_target_speed(self, target_velocity_in_scenario, steer, pred_route, pred_prob, transforms, velocities):
        target_velocity = target_velocity_in_scenario / 3.6
        sff_potential = 0.0
        final_sff = None

        if self.player != None:
            agent_tr = self.player.get_transform()
            agent_v = self.player.get_velocity()
            agent_f = agent_tr.get_forward_vector()
            agent_r = agent_tr.get_right_vector()

            agent_f += agent_r * steer
            close_npcs = []

            for i in range(self.npc_count):
                if ((agent_tr.location.x - transforms[i].location.x) ** 2 + (agent_tr.location.y - transforms[i].location.y) ** 2 ) < 32*32:
                    close_npcs.append(i)
            
            potential = [100.] * 7

            if len(close_npcs) > 0:


                self.fx_buffer = self.fx_buffer * 0.9 + agent_f.x * 0.1
                self.fy_buffer = self.fy_buffer * 0.9 + agent_f.y * 0.1
                fx = self.fx_buffer
                fy = self.fy_buffer

                px = agent_tr.location.x + fx * 3.
                py = agent_tr.location.y + fy * 3.

                for npci in close_npcs:

                    dx = transforms[npci].location.x - px
                    dy = transforms[npci].location.y - py
                    d = abs(fx * dy - fy * dx)
                    if d < 2.:
                        dist = np.sqrt(dx * dx + dy * dy)
                        if potential[0] > dist:
                            potential[0] = dist

                    for i in range(len(pred_prob[npci])):
                        if pred_prob[npci][i] > 0.01:
                            nx = transforms[npci].location.x
                            ny = transforms[npci].location.y
                            for j in range(3):
                                dx = (nx + pred_route[npci][i][j * 2]) / 2 - px
                                dy = (ny + pred_route[npci][i][j * 2 + 1]) / 2  - py
                                d = abs(fx * dy - fy * dx)
                                if d < 2.:
                                    dist = np.sqrt(dx * dx + dy * dy)
                                    dist = 100 - (100 - dist) * (pred_prob[npci][i] ** 0.5)
                                    if potential[j * 2 + 1] > dist:
                                        potential[j * 2 + 1] = dist

                                nx = pred_route[npci][i][j * 2]
                                ny = pred_route[npci][i][j * 2 + 1]
                                dx = nx - px
                                dy = ny  - py
                                d = abs(fx * dy - fy * dx)
                                if d < 2.:
                                    dist = np.sqrt(dx * dx + dy * dy)
                                    dist = 100 - (100 - dist) * (pred_prob[npci][i] ** 0.5)
                                    if potential[j * 2 + 2] > dist:
                                        potential[j * 2 + 2] = dist


                #print("prob field", potential)
                
                for i in range(7):
                    dv = potential[i] / (i * 0.5 + 1.) - 1.
                    if target_velocity > dv :
                        target_velocity = dv

                #print("target_velocity", target_velocity)
                    
                if target_velocity < 1: # HO ADDED
                    target_velocity = 0.

            if target_velocity < 0.:
                target_velocity = 0.

            self.log_file.write("\t".join([str(v) for v in potential]) + "\n")
            return target_velocity
            
        else:
            return 0


    def destroy(self):
        pass