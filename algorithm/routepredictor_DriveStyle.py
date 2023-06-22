

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
agent_for_each_train = 8
global_latent_len = 4
l2_regularizer_weight = 0.0001
global_regularizer_weight = 0.001


def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

class RoutePredictor:
    def __init__(self, laneinfo, agent_count=100):

        tf.disable_eager_execution()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, global_latent_len=global_latent_len, 
                                            l2_regularizer_weight=l2_regularizer_weight, global_regularizer_weight=global_regularizer_weight, route_len=route_len, action_len= action_len)
            learner_saver = tf.train.Saver(var_list=self.learner.trainable_dict, max_to_keep=0)
            learner_saver.restore(self.sess, "train_log/DrivingStyle8_fake/log_15-06-2023-16-56-40_10.ckpt")

        self.lane_tracers = [LaneTrace(laneinfo, 10) for _ in range(agent_count)]
        self.output_route_len = 5
        self.output_route_num = 3

    def Assign_NPCS(self, npcs):
        self.npcs = npcs
        self.agent_count = len(npcs)

    def Get_Predict_Result(self, close_npcs, npc_transforms, npc_velocities, actor_transform, actor_velocitiy, npc_traffic_sign, npc_impatience):
        state_dic = []
        route_dic = []
        action_dic = []
        for i in close_npcs:
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
            state_dic.append(state)
            state_dic.append(state)
            state_dic.append(state)
            route_dic.append(route)
            route_dic.append(route)
            route_dic.append(route)
            action_dic.extend([0, 1, 2])

        global_latent_dic = [[0., 0., 0., 0.] for _ in range(len(state_dic)) ]
        with self.sess.as_default():
            res_prob = self.learner.get_decoded_action(state_dic, route_dic, global_latent_dic)
            res_route = self.learner.get_decoded_route(state_dic, route_dic, action_dic, global_latent_dic)

        self.pred_prob = []
        self.pred_route = []
        yawsin = np.sin(tr.rotation.yaw * 0.017453293)
        yawcos = np.cos(tr.rotation.yaw * 0.017453293)
        for i in range(len(res_route)):
            l = []
            px = x
            py = y
            l.append([px, py])
            for j in range(5):
                dx, dy = rotate(res_route[i][2 * j], res_route[i][2 * j + 1], yawsin, yawcos)
                px += dx
                py += dy
                l.append([px, py])
            self.pred_route.append(l)
            self.pred_prob.append(res_prob[i][i % 3])
