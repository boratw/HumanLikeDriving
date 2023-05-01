
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
from laneinfo import LaneInfo, RouteTracer
from network.DrivingStyle3 import DrivingStyleLearner
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
    state_dic = [cur_history[target][current_step + j][0] for j in range(0, 100, 20)]
    nextstate_dic = [cur_history[target][current_step + j][1] for j in range(0, 100, 20)]
    latent_dic = [[float(list[i + 1]) for i in range(4)] for _ in range(0, 100, 20) ]
    
    with sess.as_default():
        res = learner.get_global_decoded(state_dic, nextstate_dic, latent_dic)


    d_state = [[state_vectors[current_step + j][target][0], state_vectors[current_step + j][target][1], state_vectors[current_step + j][target][2] * 0.017453293] for j in range(0, 100, 20)]

    l_state = []
    x, y = state_vectors[current_step][target][0], state_vectors[current_step][target][1]
    yawsin = np.sin(state_vectors[current_step][target][2] * 0.017453293)
    yawcos = np.cos(state_vectors[current_step][target][2] * 0.017453293)
    for j in range(len(res)):
        l_state.append([x, y, d_state[j][2]])
        px, py = rotate(res[j][0], res[j][1], yawsin, yawcos)
        x += px
        y += py
    res = json.dumps({"route" : d_state, "predicted" : l_state})

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
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")

state_len = 62
nextstate_len = 2
agent_for_each_train = 8
global_latent_len = 4

pkl_index = 0
exp_index = 0

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = DrivingStyleLearner(state_len=state_len, agent_for_each_train=agent_for_each_train, global_latent_len=global_latent_len)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    learner_saver.restore(sess, "train_log/DrivingStyle3/log_28-04-2023-15-25-37_3000.ckpt")

    with open("data/gathered_from_npc_batjeon2/data_" + str(pkl_index) + ".pkl","rb") as fr:
        data = pickle.load(fr)

    cur_history = [[] for _ in range(100)]

    state_vectors = data[exp_index]["state_vectors"]
    agent_count = len(state_vectors[0])
    step_count = len(state_vectors) - 50

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

            relposx = state_vectors[step+20][i][0] - x
            relposy = state_vectors[step+20][i][1] - y
            px, py = rotate(relposx, relposy, yawsin, yawcos)
            route = [px, py]
            
            waypoints = []
            px, py = 0., 0.
            prevx = 0.
            prevy = 0.
            k = step
            for j in range(5):
                while k < len(state_vectors):
                    if len(state_vectors[k][i][8]) > 0 :
                        if ((state_vectors[k][i][8][0][1] - prevx) ** 2 + (state_vectors[k][i][8][0][2] - prevy) ** 2) > 25:
                            relposx = state_vectors[k][i][8][0][1] - x
                            relposy = state_vectors[k][i][8][0][2] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            prevx = state_vectors[k][i][8][0][1]
                            prevy = state_vectors[k][i][8][0][2]
                            break
                    k += 1
                waypoints.extend([px, py])
                
            px, py = 50., 0.
            for t in state_vectors[step][i][6]:
                if np.sqrt(px * px + py * py) >  np.sqrt((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                    px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
                    
            cur_history[i].append( [np.concatenate([[velocity, (1. if state_vector[i][5] == 0. else 0.), px, py], waypoints, other_vcs[:8,:6].flatten()]), route, torque_added[i]])
            if torque_added[i] > 0:
                torque_added[i] -= 1

    global_latents = [[] for _ in range(agent_count)]
    global_latent_mean = np.zeros((agent_count, global_latent_len))

    for step in range(0, step_count):
        if step % 100 == 0:
            print("Getting latent from step " + str(step))
        state_dic = []
        nextstate_dic = []
        for x in range(agent_count):
            state_dic.append(cur_history[x][step][0])
            nextstate_dic.append(cur_history[x][step][1])

        res = learner.get_latent(state_dic, nextstate_dic)
        for x in range(agent_count):
            if cur_history[x][step][2] == 0:
                global_latent_mean[x] += res[x]
                global_latents[x].append(res[x])
            else:
                global_latents[x].append(np.zeros((global_latent_len, )))

    
    
    for x in range(agent_count):
        global_latent_mean[x] /= np.sqrt(np.sum(global_latent_mean[x] ** 2))
    


    with open("log_drivingstyle.txt", "wt") as f:
        for i in range(agent_count):
            f.write("\t".join([str(global_latent_mean[i][j]) for j in range(global_latent_len)]))
            f.write("\t")
            f.write("\t".join([str(j) for j in data[exp_index]["params"][i]]))
            f.write("\n")

            


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

        