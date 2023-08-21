

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

from network.DrivingStyle8 import DrivingStyleLearner

state_len = 53
nextstate_len = 10
route_len = 20
action_len = 3
global_latent_len = 4
l2_regularizer_weight = 0.0001
global_regularizer_weight = 0.001
learner_lr_start = 0.0001
learner_lr_end = 0.00001
fake_weight = 0.01

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

class RoutePredictor_DriveStyle:
    def __init__(self, laneinfo, agent_count=100, use_global_latent=False):

        tf.disable_eager_execution()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, global_latent_len=global_latent_len, 
                                            l2_regularizer_weight=l2_regularizer_weight, global_regularizer_weight=global_regularizer_weight, route_len=route_len, action_len= action_len)
            learner_saver = tf.train.Saver(var_list=self.learner.trainable_dict, max_to_keep=0)
            learner_saver.restore(self.sess, "train_log/DrivingStyle8_3/log_14-08-2023-13-41-51_240.ckpt")

        self.lane_tracers = [LaneTrace(laneinfo, 10) for _ in range(agent_count)]
        self.output_route_len = 5
        self.output_route_num = 3
        self.use_global_latent = use_global_latent

    def Assign_NPCS(self, npcs):
        self.npcs = npcs
        self.agent_count = len(npcs)

        self.global_latent_sum = np.zeros((self.agent_count, global_latent_len))
        self.global_latent_mean = np.zeros((self.agent_count, global_latent_len))
        self.global_latent_divider = np.zeros((self.agent_count, global_latent_len)) + 1e-7

        self.state_history = [[] for _ in range(self.agent_count)]
        self.pos_history = [[] for _ in range(self.agent_count)]
        self.route_history = [[] for _ in range(self.agent_count)]

        self.global_latent_parsed = [False for _ in range(self.agent_count)]
        self.global_latent_parsed_count = 0

    def Get_Predict_Result(self, close_npcs, npc_transforms, npc_velocities, actor_transform, actor_velocitiy, npc_traffic_sign, npc_impatience):
        state_dic = []
        route_dic = []
        action_dic = []
        global_latent_dic = []
        for i in range(self.agent_count):
            if i in close_npcs:
                tr = npc_transforms[i]
                v = npc_velocities[i]
                x = tr.location.x
                y = tr.location.y
                yawsin = np.sin(tr.rotation.yaw  * -0.017453293)
                yawcos = np.cos(tr.rotation.yaw  * -0.017453293)
                other_vcs = []
                for j in range(self.agent_count):
                    if i != j:
                        relposx = npc_transforms[j].location.x - x
                        relposy = npc_transforms[j].location.y - y
                        px, py = rotate(relposx, relposy, yawsin, yawcos)
                        vx, vy = rotate(npc_velocities[j].x, npc_velocities[j].y, yawsin, yawcos)
                        relyaw = (npc_transforms[j].rotation.yaw - tr.rotation.yaw)   * 0.017453293
                        other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])

                relposx = actor_transform.location.x - x
                relposy = actor_transform.location.y - y
                px, py = rotate(relposx, relposy, yawsin, yawcos)
                vx, vy = rotate(actor_velocitiy.x, actor_velocitiy.y, yawsin, yawcos)
                relyaw = (actor_transform.rotation.yaw - tr.rotation.yaw)   * 0.017453293
                other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
                while len(other_vcs) < 8:
                    other_vcs.append([100., 100., 1., 0., 0., 0., 10000.])
                other_vcs = np.array(sorted(other_vcs, key=lambda s: s[6]))

                velocity = np.sqrt(v.x ** 2 + v.y ** 2)

                traced, tracec = self.lane_tracers[i].Trace(x, y)
                route = []
                if traced == None:
                    for trace in range(action_len):
                        waypoints = []
                        for j in range(route_len // 2):
                            waypoints.extend([0., 0.])
                        route.append(waypoints)
                else:
                    for trace in traced:
                        waypoints = []
                        for j in trace:
                            px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                            waypoints.extend([px, py])
                        route.append(waypoints)

                state = np.concatenate([[velocity, (1. if npc_traffic_sign[i] == 0. else 0.), px, py, npc_impatience[i]], other_vcs[:8,:6].flatten()])
                #state = np.concatenate([[velocity, (1. if npc_traffic_sign[i] == 0. else 0.), px, py], other_vcs[:8,:6].flatten()])
                state_dic.append(state)
                state_dic.append(state)
                state_dic.append(state)
                route_dic.append(route)
                route_dic.append(route)
                route_dic.append(route)
                global_latent_dic.append(self.global_latent_mean[i])
                global_latent_dic.append(self.global_latent_mean[i])
                global_latent_dic.append(self.global_latent_mean[i])
                action_dic.extend([0, 1, 2])

                if self.use_global_latent:
                    self.state_history[i].append(state)
                    self.pos_history[i].append([x, y, yawsin, yawcos])
                    self.route_history[i].append(route)

            else:
                if self.use_global_latent:
                    self.state_history[i] = []
                    self.pos_history[i] = []
                    self.route_history[i] = []


        with self.sess.as_default():
            res_prob = self.learner.get_decoded_action(state_dic, route_dic, global_latent_dic)
            res_route = self.learner.get_decoded_route(state_dic, route_dic, action_dic, global_latent_dic)

        self.pred_prob = []
        self.pred_route = []
        for i in range(len(res_route)):
            l = []
            tr = npc_transforms[close_npcs[i // 3]]
            px = tr.location.x
            py = tr.location.y
            yawsin = np.sin(tr.rotation.yaw  * 0.017453293)
            yawcos = np.cos(tr.rotation.yaw  * 0.017453293)
            l.append([px, py])
            for j in range(5):
                dx, dy = rotate(res_route[i][2 * j], res_route[i][2 * j + 1] * 2, yawsin, yawcos)
                px += dx
                py += dy
                l.append([px, py])
            self.pred_route.append(l)
            #self.pred_prob.append(res_prob[i][i % 3])
            self.pred_prob.append(1. - (0.9 - res_prob[i][i % 3] * 0.9) ** 2)

        if self.use_global_latent:
            state_dic = []
            route_dic = []
            nextstate_dic = []
            index = []
            for i in range(self.agent_count):
                if len(self.state_history[i]) > 90:
                    state_dic.append(self.state_history[i][0])
                    route_dic.append(self.route_history[i][0])

                    nextstate = []
                    for j in range(0, 75, 15) :
                        relposx = self.pos_history[i][j + 15][0] - self.pos_history[i][j][0]
                        relposy = self.pos_history[i][j + 15][1] - self.pos_history[i][j][1]
                        px, py = rotate(relposx, relposy, self.pos_history[i][0][2], self.pos_history[i][0][3])
                        nextstate.extend([px, py])
                    nextstate_dic.append(nextstate)
                
                    self.state_history[i] = self.state_history[i][1:]
                    self.route_history[i] = self.route_history[i][1:]
                    self.pos_history[i] = self.pos_history[i][1:]
                    index.append(i)

            if len(index) > 0:
                with self.sess.as_default():
                    res_mu, res_sig = self.learner.get_latent(state_dic, nextstate_dic, route_dic)
                    for i in range(len(index)):
                        x = index[i]
                        self.global_latent_sum[x] += res_mu[i] * np.exp(-res_sig[i])
                        self.global_latent_divider[x] += np.exp(-res_sig[i])
                        self.global_latent_mean[x] = self.global_latent_sum[x] / self.global_latent_divider[x]

                        if self.global_latent_parsed[x] == False:
                            self.global_latent_parsed[x] = True
                            self.global_latent_parsed_count += 1