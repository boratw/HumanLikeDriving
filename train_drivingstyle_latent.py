
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
from network.DrivingStyle_Latent3 import DrivingStyleLearner
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import multiprocessing
import math
import copy

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

state_len = 83
prevstate_len = 6
nextstate_len = 6
agent_num = 100
action_len = 31

log_name = "train_log/DrivingStyle_Latent4_nextstate/log_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = open(log_name + ".txt", "wt")

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

ReadOption = { "LaneFollow" : [1., 0., 0.],
              "Left" : [0., 0., 1.],
              "Right" : [0., 0., -1.],
              "ChangeLaneLeft" : [0., 1., 0.],
              "ChangeLaneRight" : [0., -1, 0.],
              "Straight" : [1., 0., 0.]
              }

def parallel_task(item):

    item = copy.deepcopy(item)
    
    
    state_vectors = item["state_vectors"]
    control_vectors = item["control_vectors"]
    agent_count = len(item["state_vectors"][0])

    history_exp = [[] for _ in range(agent_count)]
    lane_tracers = [LaneTrace(laneinfo, 5) for _ in range(agent_count)]
    for step in range(120, len(state_vectors)-120):
        for i in range(agent_count):
            if ((state_vectors[step - 80][i][0] - state_vectors[step + 80][i][0]) ** 2 + \
                (state_vectors[step - 80][i][1] - state_vectors[step + 80][i][1]) ** 2) > 1:
                if control_vectors[step][i][0] == 0 and control_vectors[step+4][i][0] == 0 and \
                    control_vectors[step+12][i][0] == 0 and control_vectors[step+16][i][0] == 0 and control_vectors[step+20][i][0] == 0:

                    x = state_vectors[step][i][0]
                    y = state_vectors[step][i][1]
                    traced, tracec = lane_tracers[i].Trace(x, y)
                    if traced is not None:
                        other_vcs = []
                        yawsin = np.sin(state_vectors[step][i][2]  * -0.017453293)
                        yawcos = np.cos(state_vectors[step][i][2]  * -0.017453293)

                        velocity = math.sqrt(state_vectors[step][i][3] ** 2 + state_vectors[step][i][4] ** 2)

                        prevstate = []
                        for j in range(-60, 0, 20) :
                            relposx = state_vectors[step + j][i][0] - x
                            relposy = state_vectors[step + j][i][1] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            prevstate.extend([px, py])
                        nextstate = []
                        for j in range(20, 80, 20) :
                            relposx = state_vectors[step + j][i][0] - x
                            relposy = state_vectors[step + j][i][1] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            nextstate.extend([px, py])
                        
                        distance_array = [(state_vectors[step][j][0] - x) ** 2 + (state_vectors[step][j][1] - y) ** 2 for j in range(agent_count)]
                        distance_indicies = np.array(distance_array).argsort()
                        other_vcs = []
                        for j in distance_indicies[1:9]:
                            relposx = state_vectors[step][j][0] - x
                            relposy = state_vectors[step][j][1] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            vx, vy = rotate(state_vectors[step][j][3], state_vectors[step][j][4], yawsin, yawcos)
                            relyaw = (state_vectors[step][j][2] - state_vectors[step][i][2])   * 0.017453293
                            other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy])

                        route = []
                        for trace in traced:
                            waypoints = []
                            for j in trace:
                                px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                                waypoints.extend([px, py])
                            route.append(waypoints)


                        action = 0.
                        px, py = 0., 0.
                        for j, r in enumerate([4, 2, 1]):
                            t = (nextstate[j * 2 + 1] - px) / (nextstate[j * 2] - py + 1.)
                            if t > 0.142857143 :
                                t = 0.142857143
                            elif t < -0.142857143:
                                t = -0.142857143
                            action += t * r
                            px = nextstate[j * 2 + 1]
                            py = nextstate[j * 2]

                        px, py = 100., 0.
                        if state_vectors[step][i][5] == 0:
                            for t in state_vectors[step][i][6]:
                                if (px * px + py * py) >  ((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                                    px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
                                    
                        action = round(action * 15) + 15
                        if action < 0:
                            action = 0
                        elif action > 30:
                            action = 30
                        history_exp[i].append( [np.concatenate([[velocity, (1. if state_vectors[step][i][5] == 0. else 0.), px, py, control_vectors[step][i][1]], 
                                                                np.array(other_vcs).flatten(), np.array(route).flatten()]), prevstate, nextstate, action])
    return history_exp

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = DrivingStyleLearner(state_len=state_len, prevstate_len=prevstate_len, nextstate_len=nextstate_len, action_len=action_len)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    learner.network_initialize()
    #learner_saver.restore(sess, "train_log/DrivingStyle11_Bayesian_Latent/log_2023-09-19-18-22-43_20.ckpt")
    log_file.write("Epoch" + learner.log_caption() + "\n")
    with multiprocessing.Pool(20) as pool:
        history = []
        for epoch in range(1, 10000):
            pkl_index = random.randrange(32)
            print("Epoch " + str(epoch))
            if epoch % 20 == 1:
                history = history[(len(history) // 16):]
                print("Read data " + str(pkl_index))
                with open("data/gathered_from_npc1/data_" + str(pkl_index) + ".pkl","rb") as fr:
                    data = pickle.load(fr)

                    history_data = [[] for _ in range(agent_num)]
                    for result in pool.imap_unordered(parallel_task, data):
                        for i, r in enumerate(result):
                            history_data[i].extend(r)
                    for r in history_data:
                        if len(r) > 64:
                            history.append(r)

            print("Current History Length : " + str(len(history)))
            for iter in range(len(history) * 4):
                        
                state_dic = []
                prevstate_dic = []
                nextstate_dic = []
                action_dic = []
                agent_indices = random.sample(range(len(history)), 8)
                for agent_index in agent_indices:
                    cur_history = history[agent_index]
                    step_dic = random.sample(range(len(cur_history)), 64)

                    state_dic.extend([cur_history[step][0] for step in step_dic])
                    prevstate_dic.extend([cur_history[step][1] for step in step_dic])
                    nextstate_dic.extend([cur_history[step][2] for step in step_dic])
                    action_dic.extend([cur_history[step][3] for step in step_dic])
                learner.optimize(state_dic, prevstate_dic, nextstate_dic, action_dic)
            learner.log_print()
            log_file.write(str(epoch) + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()




            if epoch % 20 == 0:
                learner_saver.save(sess, log_name + "_" + str(epoch) + ".ckpt")

