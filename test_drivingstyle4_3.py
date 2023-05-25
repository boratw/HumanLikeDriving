
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
from network.DrivingStyle4 import DrivingStyleLearner
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
from visualizer.server import VisualizeServer
import json


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

def SendCurVehicle(target):
    i = int(target[0])
    d_latent_history = [global_latents[i][j].tolist() for j in range(step_count)]
    d_latent = global_latent_mean[i].tolist()
    res = json.dumps({"global_latent" : d_latent, "global_latent_history" : d_latent_history})
    return res


def SendLatentOutput(list):
    target = int(list[0])
        
    print("Getting Predict result of step " + str(current_step) +" of vehicle " + str(target))
    state_dic = [cur_history[target][current_step][0] for _ in range(5)]
    nextstate_dic = [cur_history[target][current_step][1] for _ in range(5)]
    global_latent_dic = [[float(list[i + 1]) for i in range(4)] for _ in range(5) ]
    for i in range(5):
        state_dic[i][4:24] = np.array(cur_history[target][current_step][3][i])
    
    
    with sess.as_default():
        res = learner.get_global_decoded(state_dic, nextstate_dic, global_latent_dic)


    d_state = [[state_vectors[current_step + j][target][0], state_vectors[current_step + j][target][1], state_vectors[current_step + j][target][2] * 0.017453293] for j in range(0, 120, 20)]
    l_state = []

    yawsin = np.sin(state_vectors[current_step][target][2] * 0.017453293)
    yawcos = np.cos(state_vectors[current_step][target][2] * 0.017453293)
    for i in range(len(res)):
        l = []
        x, y = state_vectors[current_step][target][0], state_vectors[current_step][target][1]
        l.append([x, y, d_state[0][2]])
        for j in range(5):
            px, py = rotate(res[i][2 * j], res[i][2 * j + 1], yawsin, yawcos)
            x += px
            y += py
            l.append([x, y, d_state[j][2]])
        l_state.append(l)
    res = json.dumps({"route" : d_state, "predicted" : l_state})

    print(cur_history[target][current_step][3])

    return res

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos



laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

state_len = 52 + 20
nextstate_len = 10
agent_for_each_train = 8
global_latent_len = 4
local_latent_len = 0
l2_regularizer_weight = 0.0001
global_regularizer_weight = 0.001

pkl_index = 0
exp_index = 0

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, agent_for_each_train=agent_for_each_train, global_latent_len=global_latent_len, local_latent_len= local_latent_len)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    learner_saver.restore(sess, "train_log/DrivingStyle4/log_15-05-2023-23-47-12_300.ckpt")

    with open("data/gathered_from_npc_batjeon2/data_" + str(pkl_index) + ".pkl","rb") as fr:
        data = pickle.load(fr)

    cur_history = [[] for _ in range(100)]

    state_vectors = data[exp_index]["state_vectors"]
    agent_count = len(state_vectors[0])
    step_count = len(state_vectors) - 150
    lane_tracers = [LaneTrace(laneinfo) for _ in range(agent_count)]

    torque_added = [0 for _ in range(100)]
    for step, state_vector in enumerate(state_vectors[:step_count]):
        if step % 100 == 0:
            print("Read Step " + str(step))
        for i in range(agent_count):
            if state_vectors[step+20][i][9] != 0:
                torque_added[i] = 25
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

            route = []
            for j in range(0, 100, 20) :
                relposx = state_vectors[step + j + 20][i][0] - state_vectors[step + j][i][0]
                relposy = state_vectors[step + j + 20][i][1] - state_vectors[step + j][i][1]
                px, py = rotate(relposx, relposy, yawsin, yawcos)
                route.extend([px, py])
            
            traced, tracec = lane_tracers[i].Trace(x, y)
            waypoints = []
            minj = traced[0]
            mindist = 99999
            for trace, c in zip(traced, tracec):
                if c:traced
                    dist = (trace[9][0] - state_vectors[step + 100][i][0]) ** 2 + (trace[9][1] - state_vectors[step + 100][i][1]) ** 2
                    if dist < mindist:
                        minj = trace
                        mindist = dist
            for j in minj:
                px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                waypoints.extend([px, py])
                
            true_waypoints = []
            for trace in traced:
                w = []
                for j in trace:
                    px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                    w.extend([px, py])
                true_waypoints.append(w)

            px, py = 50., 0.
            for t in state_vectors[step][i][6]:
                if np.sqrt(px * px + py * py) >  np.sqrt((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                    px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
                    
            cur_history[i].append( [np.concatenate([[velocity, (1. if state_vector[i][5] == 0. else 0.), px, py], waypoints, other_vcs[:8,:6].flatten()]), route, torque_added[i], true_waypoints])
            if torque_added[i] > 0:
                torque_added[i] -= 1

    global_latents = [[] for _ in range(agent_count)]
    local_latents = [[] for _ in range(agent_count)]
    global_latent_mean = np.zeros((agent_count, global_latent_len))

    for step in range(0, step_count):
        if step % 100 == 0:
            print("Getting latent from step " + str(step))
        state_dic = []
        nextstate_dic = []
        for x in range(agent_count):
            state_dic.append(cur_history[x][step][0])
            nextstate_dic.append(cur_history[x][step][1])

        res_global, res_local = learner.get_latent(state_dic, nextstate_dic)
        for x in range(agent_count):
            if cur_history[x][step][2] == 0:
                global_latent_mean[x] += res_global[x]
                global_latents[x].append(res_global[x])
            else:
                global_latents[x].append(np.zeros((global_latent_len, )))
            local_latents[x].append(res_local[x])

    
    
    for x in range(agent_count):
        global_latent_mean[x] /= np.sqrt(np.sum(global_latent_mean[x] ** 2))
    


    with open("log_drivingstyle.txt", "wt") as f:
        for i in range(agent_count):
            f.write("\t".join([str(global_latent_mean[i][j]) for j in range(global_latent_len)]))
            f.write("\t")
            f.write("\t".join([str(j) for j in data[exp_index]["params"][i]]))
            f.write("\n")

    lane_tracers = [LaneTrace(laneinfo) for _ in range(agent_count)]
            


    server = VisualizeServer()
    server.handlers["curstate"] = SendCurstate
    server.handlers["expinfo"] = SendExpInfo
    server.handlers["agentinfo"] = SendCurVehicle
    server.handlers["latentout"] = SendLatentOutput
    try:
        while(True):
            server.Receive()
    finally:
        server.Destroy()

        