
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
from network.TrajectoryEstimator2 import TrajectoryEstimator
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
from visualizer.server import VisualizeServer

pkl_index = 0
exp_index = 1
read_step_count = 1990
current_step = 0
latent_sampling_num = 20
global_latent_len=4
local_latent_len=4
traj_len=10

def SendCurstate(step):
    step = int(step[0])
    res = "{\"state\":[["
    state_vectors = data[exp_index]["state_vectors"]
    num_agent = len(state_vectors[step])
    for i in range(num_agent):
        res += str(state_vectors[step][i][0]) + "," + str(state_vectors[step][i][1]) + "," + str(state_vectors[step][i][2] * 0.017453293) + "],["

    res = res[:-2] + "],\"route\":[[["
    for i in range(num_agent):
        for j in range(0, 100 // traj_len + 100, 100 // traj_len):
            if step + j < read_step_count:
                res += str(state_vectors[step+j][i][0]) + "," + str(state_vectors[step+j][i][1]) + "],["
        res = res[:-2] + "],[["

    res = res[:-3] + "],\"latents\":[[["
    for i in range(num_agent):
        for j in range(global_latent_len):
            res += str(global_latent[i][0][j]) + "," + str(global_latent[i][1][j]) + "],["
        print("1")
        for j in range(local_latent_len):
            res += str(local_latent_record[i][step][0][j]) + "," + str(local_latent_record[i][step][0][j]) + "],["
        print("2")
        res = res[:-2] + "],[["

    res = res[:-3] + "]}"
    global current_step
    current_step = step
    return res

def SendExpInfo(nothing):
    res = "{\"max_step\":" + str(read_step_count) + \
        ", \"latent_len\":" + str(global_latent_len + local_latent_len) + "}"
    return res

def SendCurVehicle(step):
    i = int(step[0])
    res = "{\"global_latent_record\":[["
    for l in global_latent_record[i]:
        res += str(l[0]) + "," + str(l[1]) + "," + str(l[2]) + "," + str(l[3]) + "],["
    res = res[:-2] + "],\"local_latent_record\":[["
    for l in local_latent_record[i]:
        res += str(l[0][0]) + "," + str(l[0][1]) + "," + str(l[0][2]) + "," + str(l[0][3]) + "],["
    res = res[:-2] + "]}"
    return res


def SendLatentOutput(list):
    with sess.as_default():
        i = int(list[0])

        
        target_global_dic = [[float(list[x]) for x in range(1, 5)]]
        target_local_dic = [[float(list[x]) for x in range(5, 9)]]
        state_vector = state_vectors[current_step]
        global yawsin
        global yawcos
        
        other_vcs = []
        yawsin = np.sin(state_vector[i][2]  * -0.017453293)
        yawcos = np.cos(state_vector[i][2]  * -0.017453293)
        for j in range(agent_count):
            if i != j:
                relposx = state_vector[j][0] - state_vector[i][0]
                relposy = state_vector[j][1] - state_vector[i][1]
                px, py = rotate(relposx, relposy)
                vx, vy = rotate(state_vector[j][3], state_vector[j][4])
                relyaw = (state_vector[j][2] - state_vector[i][2])   * 0.017453293
                if relyaw < -np.pi:
                    relyaw += 2 * np.pi
                elif relyaw > np.pi:
                    relyaw -= 2 * np.pi
                other_vcs.append([relposx, relposy, relyaw, vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
        other_vcs = np.array(sorted(other_vcs, key=lambda s: s[5]))
        velocity = np.sqrt(state_vector[i][3] ** 2 + state_vector[i][4] ** 2)
        waypoints = []
        option = [0., 0., 0.]
        for j in range(3):
            if j < len(state_vector[i][7]):
                relposx = state_vector[i][7][j][1]
                relposy = state_vector[i][7][j][2]
                px, py = rotate(relposx, relposy)
                if state_vector[i][7][j][0] in ReadOption:
                    option = ReadOption[state_vector[i][7][j][0]]
                else:
                    print("Unknown RoadOption " + state_vector[i][7][j][0])
            waypoints.append([option[0], option[1], option[2], px, py])

        state_dic = [[velocity, state_vector[i][5]]]
        waypoint_dic = [waypoints]
        othervcs_dic = [other_vcs[:8, :5]]
        res = learner.get_routes(state_dic, waypoint_dic, othervcs_dic, target_global_dic, target_local_dic)

        p = []
        yawsin = np.sin(state_vector[i][2]  * 0.017453293)
        yawcos = np.cos(state_vector[i][2]  * 0.017453293)
        relposx = 0.
        relposy = 0.
        p.append([state_vector[i][0], state_vector[i][1]])
        for j in range(traj_len):
            relposx = relposx * 0.75 + abs(res[0][j][0])
            relposy = relposy * 0.75 + res[0][j][1]
            #relposx = res[0][j][0]
            #relposy = res[0][j][1]
            px, py = rotate(relposx, relposy)
            p.append([px + state_vector[i][0], py + state_vector[i][1]])
        
        res = "{\"predicted\":[["
        for k in range(traj_len):
            res += str(p[k][0]) + "," + str(p[k][1]) + "],["
        res = res[:-2] + "]}"
        return res



def rotate(posx, posy):
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
predicteds = []
tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = TrajectoryEstimator(traj_len=traj_len)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    learner_saver.restore(sess, "train_log/TrajectoryEstimator2/log_31-03-2023-13-33-14_700.ckpt")

    with open("data/gathered_from_param2_npc/data2_" + str(pkl_index) + ".pkl","rb") as fr:
        data = pickle.load(fr)

    state_vectors = data[exp_index]["state_vectors"]
    agent_count = len(data[exp_index]["state_vectors"][0])
    step_count = len(data[exp_index]["state_vectors"])


    global_latent_record = [ [] for _ in range(agent_count)]
    local_latent_record = [ [] for _ in range(agent_count)]
    global_latent = []
    local_latent = []
    for step, state_vector in enumerate(state_vectors[:read_step_count]):
        print("preload step (global) : " + str(step))
        dic = []
        for i in range(agent_count):
            other_vcs = []
            yawsin = np.sin(state_vector[i][2]  * -0.017453293)
            yawcos = np.cos(state_vector[i][2]  * -0.017453293)
            for j in range(agent_count):
                if i != j:
                    relposx = state_vector[j][0] - state_vector[i][0]
                    relposy = state_vector[j][1] - state_vector[i][1]
                    px, py = rotate(relposx, relposy)
                    vx, vy = rotate(state_vector[j][3], state_vector[j][4])
                    relyaw = (state_vector[j][2] - state_vector[i][2])   * 0.017453293
                    if relyaw < -np.pi:
                        relyaw += 2 * np.pi
                    elif relyaw > np.pi:
                        relyaw -= 2 * np.pi
                    other_vcs.append([relposx, relposy, relyaw, vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
            other_vcs = np.array(sorted(other_vcs, key=lambda s: s[5]))
            velocity = np.sqrt(state_vector[i][3] ** 2 + state_vector[i][4] ** 2)
            route = []
            if step < step_count - 100:
                for j in range(100 // traj_len, 100 // traj_len + 100, 100 // traj_len):
                    relposx = state_vectors[step+j][i][0] - state_vector[i][0]
                    relposy = state_vectors[step+j][i][1] - state_vector[i][1]
                    px, py = rotate(relposx, relposy)
                    route.append([px, py])
            waypoints = []
            option = [0., 0., 0.]
            for j in range(3):
                if j < len(state_vector[i][7]):
                    relposx = state_vector[i][7][j][1]
                    relposy = state_vector[i][7][j][2]
                    px, py = rotate(relposx, relposy)
                    if state_vector[i][7][j][0] in ReadOption:
                        option = ReadOption[state_vector[i][7][j][0]]
                    else:
                        print("Unknown RoadOption " + state_vector[i][7][j][0])
                waypoints.append([option[0], option[1], option[2], px, py])
            dic.append( [[velocity, state_vector[i][5]], waypoints, other_vcs[:8, :5], route])

        state_dic = [x[0] for x in dic]
        waypoint_dic = [x[1] for x in dic]
        othervcs_dic = [x[2] for x in dic]
        route_dic = [x[3] for x in dic]
        if step < step_count - 100:
            latent = learner.get_global_latents(state_dic, waypoint_dic, othervcs_dic, route_dic)
            for i in range(agent_count):
                global_latent_record[i].append(latent[i])

    with open("log.txt", "wt") as f:
        for i in range(agent_count):
            global_latent.append([np.mean(global_latent_record[i], axis=0), np.var(global_latent_record[i], axis=0)])
            f.write("\t".join([str(global_latent[-1][0][j]) for j in range(4)]))
            f.write("\t")
            f.write("\t".join([str(j) for j in data[exp_index]["params"][i]]))
            f.write("\n")

              
    global_latent_dic = [x[0] for x in global_latent]
    for step, state_vector in enumerate(state_vectors[:read_step_count]):
        print("preload step (local) : " + str(step))
        dic = []
        for i in range(agent_count):
            other_vcs = []
            yawsin = np.sin(state_vector[i][2]  * -0.017453293)
            yawcos = np.cos(state_vector[i][2]  * -0.017453293)
            for j in range(agent_count):
                if i != j:
                    relposx = state_vector[j][0] - state_vector[i][0]
                    relposy = state_vector[j][1] - state_vector[i][1]
                    px, py = rotate(relposx, relposy)
                    vx, vy = rotate(state_vector[j][3], state_vector[j][4])
                    relyaw = (state_vector[j][2] - state_vector[i][2])   * 0.017453293
                    if relyaw < -np.pi:
                        relyaw += 2 * np.pi
                    elif relyaw > np.pi:
                        relyaw -= 2 * np.pi
                    other_vcs.append([relposx, relposy, relyaw, vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
            other_vcs = np.array(sorted(other_vcs, key=lambda s: s[5]))
            velocity = np.sqrt(state_vector[i][3] ** 2 + state_vector[i][4] ** 2)
            route = []
            if step < step_count - 100:
                for j in range(100 // traj_len, 100 // traj_len + 100, 100 // traj_len):
                    relposx = state_vectors[step+j][i][0] - state_vector[i][0]
                    relposy = state_vectors[step+j][i][1] - state_vector[i][1]
                    px, py = rotate(relposx, relposy)
                    route.append([px, py])
            waypoints = []
            option = [0., 0., 0.]
            for j in range(3):
                if j < len(state_vector[i][7]):
                    relposx = state_vector[i][7][j][1]
                    relposy = state_vector[i][7][j][2]
                    px, py = rotate(relposx, relposy)
                    if state_vector[i][7][j][0] in ReadOption:
                        option = ReadOption[state_vector[i][7][j][0]]
                    else:
                        print("Unknown RoadOption " + state_vector[i][7][j][0])
                waypoints.append([option[0], option[1], option[2], px, py])
            dic.append( [[velocity, state_vector[i][5]], waypoints, other_vcs[:8, :5], route])

        state_dic = [x[0] for x in dic]
        waypoint_dic = [x[1] for x in dic]
        othervcs_dic = [x[2] for x in dic]
        route_dic = [x[3] for x in dic]
        if step < step_count - 100:
            mu, var = learner.get_local_latents(state_dic, waypoint_dic, othervcs_dic, route_dic, global_latent_dic)
            for i in range(agent_count):
                local_latent_record[i].append([mu[i], var[i]])


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

        