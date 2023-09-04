import glob
import os
import sys

try:
    sys.path.append(glob.glob('/home/user/carla-0.9.14/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import tensorflow.compat.v1 as tf
from laneinfo import LaneInfo
from lanetrace import LaneTrace
from network.DrivingStyle2_bayesian_default import DrivingStyleLearner
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
from visualizer.server import VisualizeServer
import json

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



def SendCurstate(step):
    step = int(step[0])
    state_vectors = data[exp_index]["state_vectors"]
    d_state = [[state_vectors[step][i][0], state_vectors[step][i][1], state_vectors[step][i][2] * 0.017453293] for  i in range(agent_count)]
    res = json.dumps({"state" : d_state})

    global current_step
    current_step = step
    return res

def SendExpInfo(nothing):
    res = "{\"max_step\":" + str(step_count) + \
        ", \"latent_len\":" + str(global_latent_len) + "}"
    return res

def SendOutput(list):
    target = int(list[0])
        
    print("Getting Predict result of step " + str(current_step) +" of vehicle " + str(target))
    d_state = [[state_vectors[current_step + j][target][0], state_vectors[current_step + j][target][1]] for j in range(0, 90, 15)]
    l_state = [ [[0., 0., 0., 0., 0., 0. ]] for j in range(action_len)]
    o_predicted = [ [] for j in range(action_len)]
    o_action = [0.] * action_len

    for i in range(action_len):
        x = state_vectors[current_step][target][0]
        y = state_vectors[current_step][target][1]
        for step in range(5):
            state_dic = [cur_history[target][current_step + step * 15][0] for _ in range(16)]
            
            yawsin = np.sin(state_vectors[current_step + step * 15][target][2] * 0.017453293)
            yawcos = np.cos(state_vectors[current_step + step * 15][target][2] * 0.017453293)
            traced, tracec = lane_tracers[target].Trace(x, y)


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
            route_dic = [route for _ in range(16)]
            
            with sess.as_default():
                res_action, res_mean, res_var = learner.get_output(state_dic, route_dic)
            if step == 0:
                o_action[i] = float(np.mean(res_action[:, i]))
                
            o_mean =  np.mean(res_mean[:, i, :], axis=0)
            o_ep_var = np.mean(res_var[:, i, :], axis=0)
            o_al_var = np.var(res_mean[:, i, :], axis=0)

            lx, ly = 0., 0.
            for j in range(3):
                if len(o_predicted[i]) > (step + j):
                    o_predicted[i][step + j][0] = (lx + o_predicted[i][step + j][0] + o_mean[j * 2]) / 2
                    o_predicted[i][step + j][1] = (ly + o_predicted[i][step + j][1] + o_mean[j * 2 + 1]) / 2
                    o_predicted[i][step + j][2] = (o_predicted[i][step + j][2] + o_ep_var[j * 2]) / 2
                    o_predicted[i][step + j][3] = (o_predicted[i][step + j][3] + o_ep_var[j * 2 + 1]) / 2
                    o_predicted[i][step + j][4] = (o_predicted[i][step + j][4] + o_al_var[j * 2]) / 2
                    o_predicted[i][step + j][5] = (o_predicted[i][step + j][5] + o_al_var[j * 2 + 1]) / 2
                else:
                    o_predicted[i].append([o_mean[j * 2], o_mean[j * 2 + 1], o_ep_var[j * 2], o_ep_var[j * 2 + 1], o_mean[j * 2], o_mean[j * 2 + 1]])
                lx += o_mean[j * 2]
                ly += o_mean[j * 2 + 1]

            px, py = rotate(o_predicted[i][step][0], o_predicted[i][step][1], yawsin, yawcos)
            x += px
            y += py

        for step in range(5):
            l_state[i].append([float(o_predicted[i][step][0]), float(o_predicted[i][step][1]), float(np.sqrt(o_predicted[i][step][2])),
                               float(np.sqrt(o_predicted[i][step][3])), float(np.sqrt(o_predicted[i][step][4])), float(np.sqrt(o_predicted[i][step][5]))])
    res = json.dumps({"route" : d_state, "predicted" : l_state, "action_prob" : o_action}, cls=MyEncoder)
    
    return res

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos


ReadOption = { "LaneFollow" : [1., 0., 0.],
              "Left" : [0., 0., 1.],
              "Right" : [0., 0., -1.],
              "ChangeLaneLeft" : [0., 1., 0.],
              "ChangeLaneRight" : [0., -1, 0.],
              "Straight" : [1., 0., 0.]
              }

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

state_len = 53
nextstate_len = 6
route_len = 20
action_len = 3
global_latent_len = 4


pkl_index = 0
exp_index = 0

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, route_len=route_len, action_len= action_len)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    learner_saver.restore(sess, "train_log/DrivingStyle2_Bayesian_Default_2/log_27-08-2023-16-31-41_180.ckpt")

    with open("data/gathered_from_npc_batjeon6/data_" + str(pkl_index) + ".pkl","rb") as fr:
        data = pickle.load(fr)


    state_vectors = data[exp_index]["state_vectors"]
    control_vectors = data[exp_index]["control_vectors"]
    agent_count = len(state_vectors[0])
    step_count = 100 #len(state_vectors) - 150
    lane_tracers = [LaneTrace(laneinfo, 10) for _ in range(agent_count)]
    cur_history = [[] for _ in range(agent_count)]
    torque_added = [0 for _ in range(agent_count)]
    for step, state_vector in enumerate(state_vectors[100:100+step_count]):
        if step % 100 == 0:
            print("Read Step " + str(step))
        for i in range(agent_count):
            if control_vectors[step+20][i][0] != 0:
                torque_added[i] = 20
            other_vcs = []
            x = state_vectors[step][i][0]
            y = state_vectors[step][i][1]
            yawsin = np.sin(state_vectors[step][i][2]  * -0.017453293)
            yawcos = np.cos(state_vectors[step][i][2]  * -0.017453293)
            for j in range(agent_count):
                if i != j:
                    relposx = state_vectors[step][j][0] - x
                    relposy = state_vectors[step][j][1] - y
                    px, py = rotate(relposx, relposy, yawsin, yawcos)
                    vx, vy = rotate(state_vectors[step][j][3], state_vectors[step][j][4], yawsin, yawcos)
                    relyaw = (state_vectors[step][j][2] - state_vectors[step][i][2])   * 0.017453293
                    other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
            other_vcs = np.array(sorted(other_vcs, key=lambda s: s[6]))
            velocity = np.sqrt(state_vectors[step][i][3] ** 2 + state_vectors[step][i][4] ** 2)

            nextstate = []  
            for j in range(0, 45, 15) :
                relposx = state_vectors[step + j + 15][i][0] - state_vectors[step + j][i][0]
                relposy = state_vectors[step + j + 15][i][1] - state_vectors[step + j][i][1]

                px, py = rotate(relposx, relposy, yawsin, yawcos)
                nextstate.extend([px, py]) 
                
            traced, tracec = lane_tracers[i].Trace(x, y)

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
                    
            px, py = 50., 0.
            for t in state_vectors[step][i][6]:
                if np.sqrt(px * px + py * py) >  np.sqrt((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                    px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
                    
            cur_history[i].append( [np.concatenate([[velocity, (1. if state_vector[i][5] == 0. else 0.), px, py, control_vectors[step][i][1]], other_vcs[:8,:6].flatten()]), nextstate, route, torque_added[i]])
            if torque_added[i] > 0:
                torque_added[i] -= 1


    server = VisualizeServer()
    server.handlers["curstate"] = SendCurstate
    server.handlers["expinfo"] = SendExpInfo
    server.handlers["predicted"] = SendOutput
    try:
        while(True):
            server.Receive()
    finally:
        server.Destroy()