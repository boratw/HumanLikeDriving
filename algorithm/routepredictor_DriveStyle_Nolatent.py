

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

from network.DrivingStyle_Nolatent import DrivingStyleLearner

state_len = 83
prevstate_len = 6
nextstate_len = 6
agent_num = 100
action_len = 31
global_latent_len = 4
pred_num = 31

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

class RoutePredictor_DriveStyle_Nolatent:
    def __init__(self, laneinfo, npc_count = 100, player_count=1, sess=None, name="", snapshot=""):
        self.npc_count = npc_count
        self.player_count = player_count
        tf.disable_eager_execution()
        if sess == None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        with self.sess.as_default():
            self.learner = DrivingStyleLearner(name=name, state_len=state_len, nextstate_len=nextstate_len, isTraining=False)
            learner_saver = tf.train.Saver(var_list=self.learner.trainable_dict, max_to_keep=0)
            learner_saver.restore(self.sess, snapshot)

        self.lane_tracers = [LaneTrace(laneinfo, 5) for _ in range(self.npc_count)]

        self.global_latent_mean = np.zeros((self.npc_count, global_latent_len))
        self.global_latent_divider = np.zeros((self.npc_count, global_latent_len)) + 1e-7
        self.history = []
        self.use_global_latent = False

    def Assign_NPCS(self, npcs):
        self.npcs = npcs


    def Reset(self, use_global_latent=False):
        self.global_latent_mean = np.zeros((self.npc_count, global_latent_len))
        self.global_latent_divider = np.zeros((self.npc_count, global_latent_len)) + 1e-7
        self.history = []
        self.use_global_latent = use_global_latent
        pass

    def Get_Predict_Result(self, transforms, velocities,  npc_lights, impatiences):
        state_dic = []
        position_dic = []
        for i in range(self.npc_count):
            tr = transforms[i]
            v = velocities[i]
            x = tr.location.x
            y = tr.location.y
            yawsin = np.sin(tr.rotation.yaw  * -0.017453293)
            yawcos = np.cos(tr.rotation.yaw  * -0.017453293)
            other_vcs = []
            for j in range(self.npc_count + self.player_count):
                if i != j:
                    relposx = transforms[j].location.x - x
                    relposy = transforms[j].location.y - y
                    px, py = rotate(relposx, relposy, yawsin, yawcos)
                    vx, vy = rotate(velocities[j].x, velocities[j].y, yawsin, yawcos)
                    relyaw = (transforms[j].rotation.yaw - tr.rotation.yaw)   * 0.017453293
                    other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy, relposx * relposx + relposy * relposy])

            other_vcs = np.array(sorted(other_vcs, key=lambda s: s[6]))

            velocity = np.sqrt(v.x ** 2 + v.y ** 2)

            traced, tracec = self.lane_tracers[i].Trace(x, y)
            route = []
            if traced == None:
                for trace in range(3):
                    waypoints = []
                    for j in range(5):
                        waypoints.extend([0., 0.])
                    route.append(waypoints)
            else:
                for trace in traced:
                    waypoints = []
                    for j in trace:
                        px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                        waypoints.extend([px, py])
                    route.append(waypoints)
            
            if npc_lights[i][2] == 0:
                px, py = 100, 0
            else:
                px, py = rotate(npc_lights[i][1] - x, npc_lights[i][2] - y, yawsin, yawcos)

            state = np.concatenate([[velocity, npc_lights[i][0], px, py, impatiences[i]], other_vcs[:8,:6].flatten(), np.array(route).flatten()])
            position = [x, y, yawsin, yawcos]
            state_dic.append(state)
            position_dic.append(position)

        self.history.append([state_dic, position_dic])

        with self.sess.as_default():
            res_route, res_prob  = self.learner.get_output(state_dic, discrete=True)
