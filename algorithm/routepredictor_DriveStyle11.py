

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

from network.DrivingStyle11_bayesian_latent import DrivingStyleLearner

state_len = 63
nextstate_len = 6
route_len = 16
action_len = 3
global_latent_len = 4
num_of_agents = 4


def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

class RoutePredictor_DriveStyle:
    def __init__(self, laneinfo, agent_count=100):

        tf.disable_eager_execution()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, route_len=route_len, action_len= action_len, istraining=False)
            learner_saver = tf.train.Saver(var_list=self.learner.trainable_dict, max_to_keep=0)
            learner_saver.restore(self.sess, "train_log/DrivingStyle11_3_Bayesian_Latent/log_2023-11-21-17-49-04_140.ckpt")

        self.lane_tracers = [LaneTrace(laneinfo, 8) for _ in range(agent_count)]
        self.output_route_len = 3
        self.output_route_num = 3

    def Assign_NPCS(self, npcs, params):
        self.npcs = npcs
        self.agent_count = len(npcs)
        self.params = params


    def Get_Predict_Result(self, close_npcs, npc_transforms, npc_velocities, npc_control, actor_transform, actor_velocitiy, npc_traffic_sign):
        state_dic = []
        route_dic = []
        tracec_dic = []
        for i in close_npcs:
            tr = npc_transforms[i]
            v = npc_velocities[i]
            x = tr.location.x
            y = tr.location.y
            yawsin = np.sin(tr.rotation.yaw  * -0.017453293)
            yawcos = np.cos(tr.rotation.yaw  * -0.017453293)
            other_vcs = []
            for j in close_npcs:
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
            tracec_dic.append(tracec)
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

            
                    
            state = np.concatenate([[velocity], npc_traffic_sign[i], npc_control[i], self.params[i], other_vcs[:8,:6].flatten()])
            state_dic.append(state)
            route_dic.append(route)

        global_latent_dic = []
        for _ in range(len(close_npcs)):
            global_latent_dic.append(np.zeros((global_latent_len,)))


        with self.sess.as_default():
            res_route_mean, res_route_var, res_action = self.learner.get_output(state_dic, route_dic, global_latent_dic)

        pred_prob = []
        pred_route = []
        for i in range(len(close_npcs)):
            o_action = []
            for k in range(action_len):
                l = []
                tr = npc_transforms[close_npcs[i]]
                px = tr.location.x
                py = tr.location.y
                yawsin = np.sin(tr.rotation.yaw  * 0.017453293)
                yawcos = np.cos(tr.rotation.yaw  * 0.017453293)
                l.append([px, py])
                for j in range(self.output_route_len ):
                    dx, dy = rotate(res_route_mean[i][k][2 * j], res_route_mean[i][k][2 * j + 1], yawsin, yawcos)
                    px += dx
                    py += dy
                    l.append([px, py])
                pred_route.append(l)
                o_action.append(res_action[i][k])
            if tracec_dic[i][1] == False :
                o_action[0] += o_action[1]
                o_action[1] = 0.
            if tracec_dic[i][2] == False :
                o_action[0] += o_action[2]
                o_action[2] = 0.
            for k in range(action_len):
                pred_prob.append(o_action[k])

        return pred_prob, pred_route
