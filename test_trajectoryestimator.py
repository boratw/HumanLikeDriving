
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
from network.TrajectoryEstimator import TrajectoryEstimator
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
latent_len=8
traj_len=5

def SendCurstate(step):
    step = int(step[0])
    res = "{\"state\":[["
    state_vectors = data[exp_index]["state_vectors"]
    num_agent = len(state_vectors[step])
    for i in range(num_agent):
        res += str(state_vectors[step][i][0]) + "," + str(state_vectors[step][i][1]) + "," + str(state_vectors[step][i][2] * 0.017453293) + "],["

    res = res[:-2] + "],\"route\":[[["
    for i in range(num_agent):
        for j in range(0, 120, 20):
            if step - j >= 0:
                res += str(state_vectors[step-j][i][0]) + "," + str(state_vectors[step-j][i][1]) + "],["
        res = res[:-2] + "],[["

    res = res[:-3] + "],\"latent\":[[["
    for i in range(num_agent):
        for j in range(latent_len):
            res += str(latents[step][i][0][j]) + "," + str(latents[step][i][1][j]) + "],["
        res = res[:-2] + "],[["

    res = res[:-3] + "],\"predicted\":[[[["
    for i in range(num_agent):
        for j in range(latent_sampling_num):
            for k in range(5):
                res += str(predicteds[step][i][j][k][0]) + "," + str(predicteds[step][i][j][k][1]) + "],["
            res = res[:-2] + "],[["
        res = res[:-3] + "],[[["

    res = res[:-4] + "]}"
    global current_step
    current_step = step
    return res

def SendExpInfo(nothing):
    res = "{\"max_step\":" + str(read_step_count) + \
        ", \"latent_len\":" + str(latent_len) + "}"
    return res

def SendLatentOutput(list):
    if current_step >= 100:
        with sess.as_default():
            i = int(list[0])
            target_dic = [[float(list[x + 1]) for x in range(latent_len)]]
            state_vector = state_vectors[current_step]
            global yawsin
            global yawcos

            traced = routetracers[i].Trace(state_vector[i][0], state_vector[i][1], state_vector[i][2])
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

            state_dic = [[velocity, state_vector[i][5]]]
            route_dic = [(traced if traced != None else np.zeros((3, 10), dtype=np.float32))]
            othervcs_dic = [other_vcs[:8, :5]]
            res = learner.get_routes(state_dic, route_dic, othervcs_dic, target_dic)

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

    else:
        return "{\"predicted\":[[]]}"


def rotate(posx, posy):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")
routetracers = [RouteTracer(laneinfo) for _ in range(50)]
latents = []
predicteds = []
tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = TrajectoryEstimator(regularizer_weight=0.01, latent_len=latent_len, traj_len=traj_len, use_regen_loss=True)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    learner_saver.restore(sess, "train_log/TrajectoryEstimator/latent8_param7/log_17-03-2023-18-10-05_3500.ckpt")

    with open("data/gathered_from_default_npc2/data_" + str(pkl_index) + ".pkl","rb") as fr:
        data = pickle.load(fr)

    state_vectors = data[exp_index]["state_vectors"]
    agent_count = len(data[exp_index]["state_vectors"][0])
    step_count = len(data[exp_index]["state_vectors"])
    for step, state_vector in enumerate(state_vectors[:read_step_count]):
        print("preload step : " + str(step))
        dic = []
        for i in range(agent_count):
            traced = routetracers[i].Trace(state_vector[i][0], state_vector[i][1], state_vector[i][2])
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
            dic.append( [[velocity, state_vector[i][5]], (traced if traced != None else np.zeros((3, 10), dtype=np.float32)), other_vcs[:8, :5], route])

        state_dic = [x[0] for x in dic]
        route_dic = [x[1] for x in dic]
        othervcs_dic = [x[2] for x in dic]
        latent = []
        if step < step_count - 100:
            target_dic = [x[3] for x in dic]
            mu, var = learner.get_latents(state_dic, route_dic, othervcs_dic, target_dic)
            mu = np.clip(mu, -4, 4)
            var = var ** 0.8
            for i in range(agent_count):
                latent.append([mu[i], var[i]])
        else:
            for i in range(agent_count):
                latent.append([0., 1.])
        latents.append(latent)
        predicted = [[] for _ in range(agent_count)]
        if step >= 100:
            for k in range(latent_sampling_num):
                target_dic = [np.random.normal(x[0], x[1] * 10) for x in latents[-100]]
                res = learner.get_routes(state_dic, route_dic, othervcs_dic, target_dic)
                for i in range(agent_count):
                    p = []
                    yawsin = np.sin(state_vector[i][2]  * 0.017453293)
                    yawcos = np.cos(state_vector[i][2]  * 0.017453293)
                    relposx = 0.
                    relposy = 0.
                    p.append([state_vector[i][0], state_vector[i][1]])
                    for j in range(traj_len):
                        relposx = relposx * 0.75 + abs(res[i][j][0])
                        relposy = relposy * 0.75 + res[i][j][1]
                        #relposx = res[i][j][0]
                        #relposy = res[i][j][1]
                        px, py = rotate(relposx, relposy)
                        p.append([px + state_vector[i][0], py + state_vector[i][1]])
                    predicted[i].append(p)
        else:
            for k in range(latent_sampling_num):
                for i in range(agent_count):
                    predicted[i].append(np.zeros((5, 2), dtype=np.float32))
        predicteds.append(predicted)




    server = VisualizeServer()
    server.handlers["curstate"] = SendCurstate
    server.handlers["expinfo"] = SendExpInfo
    server.handlers["latentout"] = SendLatentOutput
    try:
        while(True):
            server.Receive()
    finally:
        server.Destroy()

        