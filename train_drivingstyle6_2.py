
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
from network.DrivingStyle6 import DrivingStyleLearner
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import multiprocessing

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

state_len = 53 + 6
nextstate_len = 2
agent_for_each_train = 8
global_latent_len = 4
l2_regularizer_weight = 0.0001
global_regularizer_weight = 0.01


log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/DrivingStyle6/log_" + log_name + ".txt", "wt")

ReadOption = { "LaneFollow" : 0,
              "Left" : 1,
              "Right" : 2,
              "ChangeLaneLeft" : 3,
              "ChangeLaneRight" : 4,
              "Straight" : 0
              }

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

def parallel_task(item):
    history_exp = [[] for _ in range(100)]

    state_vectors = item["state_vectors"]
    control_vectors = item["control_vectors"]
    agent_count = len(item["state_vectors"][0])

    torque_added = [0 for _ in range(100)]
    stepstart = random.randrange(50, 60)
    lane_tracers = [LaneTrace(laneinfo, 3) for _ in range(agent_count)]
    for step in range(stepstart, len(state_vectors)-150, 4):
        for i in range(agent_count):
            if torque_added[i] == 0:
                if control_vectors[step+20][i][0] != 0 or control_vectors[step+21][i][0] != 0 or control_vectors[step+22][i][0] != 0 or control_vectors[step+23][i][0] != 0:
                    torque_added[i] = 5
                else:
                    other_vcs = []
                    x = state_vectors[step][i][0]
                    y = state_vectors[step][i][1]
                    relposx = state_vectors[step+10][i][0] - x
                    relposy = state_vectors[step+10][i][1] - y
                    if (relposx * relposx + relposy * relposy) > 0.01 or random.random() < 0.1:
                        traced, tracec = lane_tracers[i].Trace(x, y)
                        if traced != None:

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

                            relposx = state_vectors[step + 20][i][0] - state_vectors[step][i][0]
                            relposy = state_vectors[step + 20][i][1] - state_vectors[step][i][1]
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            route = [px, py]
                                
                            waypoints = []
                            trace_result = traced[0]
                            if len(state_vectors[step][i][8]) >= 1:
                                mindist = 99999
                                for trace, c in zip(traced, tracec):
                                    if c:
                                        dist = (trace[1][0] - state_vectors[step][i][8][0][1]) ** 2 + (trace[1][1] - state_vectors[step][i][8][0][2]) ** 2
                                        if dist < mindist:
                                            trace_result = trace
                                            mindist = dist
                            for j in trace_result:
                                px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                                waypoints.extend([px, py])

                            
                                
                            px, py = 50., 0.
                            for t in state_vectors[step][i][6]:
                                if (px * px + py * py) >  ((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                                    px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
                            history_exp[i].append( [np.concatenate([[velocity, (1. if state_vectors[step][i][5] == 0. else 0.), px, py, control_vectors[step][i][1]], waypoints, other_vcs[:8,:6].flatten()]), route])
            else:
                torque_added[i] -= 1
    history = []
    for exp in history_exp:
        if len(exp) > 200:
            history.append(exp)
    return history

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    with multiprocessing.Pool(processes=50) as pool:
        learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, agent_for_each_train=agent_for_each_train, global_latent_len=global_latent_len, 
                                      l2_regularizer_weight=l2_regularizer_weight, global_regularizer_weight=global_regularizer_weight)
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        sess.run(tf.global_variables_initializer())
        #teacher_dict = { k:learner.trainable_dict[k] for k in learner.trainable_dict if not "Global" in k}
        #teacher_saver = tf.train.Saver(var_list=teacher_dict, max_to_keep=0)
        #learner_saver.restore(sess, "train_log/DrivingStyle4/log_02-05-2023-17-05-00/log_02-05-2023-17-05-00_900.ckpt")
        learner.network_initialize()
        log_file.write("Epoch" + learner.log_caption() + "\n")

        history = []

        for epoch in range(1, 10000):
            pkl_index = random.randrange(23)
            with open("data/gathered_from_npc_batjeon3/data_" + str(pkl_index) + ".pkl","rb") as fr:
                data = pickle.load(fr)
            print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

            history_data = []
            for result in pool.imap(parallel_task, data):
                history_data.append(result)
            history.append(history_data)

            print("Current History Length : " + str(len(history)))
            for iter in range(len(history) * 32):

                data_index = random.randrange(len(history))
                exp_index = random.randrange(len(history[data_index]))
                if iter % 32 == 31:
                    print("Train Step #" + str(iter) + "Read data " + str(data_index) + " exp " + str(exp_index))

                cur_history = history[data_index][exp_index]
                agent_num = len(cur_history)
                
                agent_dic = random.choices(list(range(agent_num)), k=agent_for_each_train)
                step_dic = [ random.choices(list(range(len(cur_history[x]))), k = 128) for x in agent_dic ]

                state_dic = []
                nextstate_dic = []
                for x in range(agent_for_each_train):
                    state_dic.extend([cur_history[agent_dic[x]][step][0] for step in step_dic[x]])
                    nextstate_dic.extend([cur_history[agent_dic[x]][step][1] for step in step_dic[x]])
                learner.optimize(epoch, state_dic, nextstate_dic)
        
                
            if len(history) > 32:
                history = history[1:]

            learner.log_print()
            log_file.write(str(epoch) + "\t" + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()


            if epoch % 50 == 0:
                learner_saver.save(sess, "train_log/DrivingStyle6/log_" + log_name + "_" + str(epoch) + ".ckpt")

