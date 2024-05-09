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
from network.DrivingStyle_Nolatent import DrivingStyleLearner as DrivingStyleLearner0
from network.DrivingStyle_Latent3 import DrivingStyleLearner as DrivingStyleLearner1
from network.DrivingStyle_Latent3_latentinput import DrivingStyleLearner as DrivingStyleLearner2
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla

module_n = int(sys.argv[1])

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")


def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos


state_len = 83
prevstate_len = 6
nextstate_len = 6
agent_num = 100
action_len = 31
global_latent_len = 4
pred_num = 31

pkl_index = 0
exp_index = 1

tf.disable_eager_execution()
sess = tf.Session()

log_txt = open("test_log/module" + str(module_n) + ".txt", "wt")

with sess.as_default():
    if module_n == 0:
        learner = DrivingStyleLearner0(state_len=state_len, nextstate_len=nextstate_len, isTraining=False)
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        learner_saver.restore(sess, "train_log/DrivingStyle_NoLatent2/log_2023-12-05-20-11-44_780.ckpt")
    elif module_n <= 3:
        learner = DrivingStyleLearner1(state_len=state_len, prevstate_len=prevstate_len, nextstate_len=nextstate_len, 
                                       isTraining=False, test_mask_use = (module_n == 3))
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        learner_saver.restore(sess, "train_log/DrivingStyle_Latent4_nextstate2/log_2023-12-28-14-55-59_1000.ckpt")
    elif module_n <= 6:
        learner = DrivingStyleLearner2(state_len=state_len, prevstate_len=prevstate_len, nextstate_len=nextstate_len, 
                                       isTraining=False, test_mask_use = (module_n == 6))
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        learner_saver.restore(sess, "train_log/DrivingStyle_AvgLatent/2024-02-16-17-42-11_600.ckpt")

    for pkl_index in range(4):
        with open("data/gathered_from_npc1/data_" + str(pkl_index) + ".pkl","rb") as fr:
            data = pickle.load(fr)

        prob_res = np.zeros(100)
        prob_var = np.zeros(100)
        close_route_res = np.zeros((100, nextstate_len))
        maximum_route_res = np.zeros((100, nextstate_len))
        close_route_var = np.zeros((100, nextstate_len))
        maximum_route_var = np.zeros((100, nextstate_len))
        res_num = 0

        #for exp_index in range(len(data)):
        for exp_index in range(10):
            print("Pkl " + str(pkl_index) + " Exp " + str(exp_index))
            state_vectors = data[exp_index]["state_vectors"]
            control_vectors = data[exp_index]["control_vectors"]
            param_vectors = data[exp_index]["params"]
            agent_count = len(state_vectors[0])
            lane_tracers = [LaneTrace(laneinfo, 5) for _ in range(agent_count)]
            cur_history = [[] for _ in range(agent_count)]

            step_start_index = 200
            step_count = len(state_vectors) - step_start_index - 150

            for step in range(step_start_index, step_start_index+step_count, 5):
                if step % 100 == 0:
                    print("Read Step " + str(step))
                for i in range(agent_count):
                    x = state_vectors[step][i][0]
                    y = state_vectors[step][i][1]
                    yawsin = np.sin(state_vectors[step][i][2]  * -0.017453293)
                    yawcos = np.cos(state_vectors[step][i][2]  * -0.017453293)
                    velocity = np.sqrt(state_vectors[step][i][3] ** 2 + state_vectors[step][i][4] ** 2)

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


                    nextstate = []  
                    for j in range(20, 80, 20) :
                        relposx = state_vectors[step + j][i][0] - x
                        relposy = state_vectors[step + j][i][1] - y
                        px, py = rotate(relposx, relposy, yawsin, yawcos)
                        nextstate.extend([px, py])
                                
                        
                    traced, tracec = lane_tracers[i].Trace(x, y)

                    route = []
                    if traced == None:
                        for trace in range(3):
                            waypoints = []
                            for j in range(5):
                                waypoints.extend([0., 0.])
                            route.append(waypoints)
                    else:
                        for trace in traced:
                            waypoints = []
                            for j in trace:
                                px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                                waypoints.extend([px, py])
                            route.append(waypoints)
                            
                    px, py = 100., 0.
                    if state_vectors[step][i][5] == 0:
                        for t in state_vectors[step][i][6]:
                            if (px * px + py * py) >  ((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                                px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
                            

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

                    action = round(action * 15) + 15
                    if action < 1:
                        action = 1
                    elif action > 29:
                        action = 29
                    cur_history[i].append( [np.concatenate([[velocity, (1. if state_vectors[step][i][5] == 0. else 0.), px, py, control_vectors[step][i][1]],
                                                            np.array(other_vcs).flatten(), np.array(route).flatten()]), 
                                            nextstate, action])


            if module_n == 0:
                for step in range(step_count // 5):
                    if step % 100 == 0:
                        print("Parsing step " + str(step))
                    state_dic = []
                    nextstate_dic = []
                    action_dic = []
                    for x in range(agent_count):
                        state_dic.append(cur_history[x][step][0])
                        nextstate_dic.append(cur_history[x][step][1])
                        action_dic.append(cur_history[x][step][2])

                    res_route, res_action = learner.get_output(state_dic, discrete=True)
                    
                    
                    for x in range(agent_count):
                        prob = (res_action[x][action_dic[x] - 1] + res_action[x][action_dic[x]] + res_action[x][action_dic[x] + 1]) / 3.
                        prob_res[x] += prob
                        prob_var[x] += prob ** 2
                        d_route = res_route[x] - np.reshape(nextstate_dic[x], (1, nextstate_len))
                        close_route_res[x] += np.abs(d_route[action_dic[x]])
                        close_route_var[x] += d_route[action_dic[x]] ** 2
                        min_route = np.argmin(np.sum(d_route ** 2, axis=1))
                        maximum_route_res[x] += np.abs(d_route[min_route])
                        maximum_route_var[x] += d_route[min_route] ** 2
                    res_num += 1
            else:
                global_latent = np.zeros((agent_count, global_latent_len))
                if module_n != 1 and module_n != 4:
                    for step in range(step_count // 5):
                        if step % 100 == 0:
                            print("Getting latent from step " + str(step))
                        state_dic = []
                        nextstate_dic = []
                        for x in range(agent_count):
                            state_dic.append(cur_history[x][step][0])
                            nextstate_dic.append(cur_history[x][step][1])

                        res_mu = learner.get_latent(state_dic, nextstate_dic, discrete=True)
                        for x in range(agent_count):
                            global_latent += res_mu[x]
                    
                    global_latent /= (step_count // 5)

                for step in range(step_count // 5):
                    if step % 100 == 0:
                        print("Parsing step " + str(step))
                    state_dic = []
                    nextstate_dic = []
                    action_dic = []
                    for x in range(agent_count):
                        state_dic.append(cur_history[x][step][0])
                        nextstate_dic.append(cur_history[x][step][1])
                        action_dic.append(cur_history[x][step][2])


                    res_route, res_action, _ = learner.get_output(state_dic, global_latent, discrete=True)
                    
                    
                    for x in range(agent_count):
                        prob = (res_action[x][action_dic[x] - 1] + res_action[x][action_dic[x]] + res_action[x][action_dic[x] + 1]) / 3.
                        prob_res[x] += prob
                        prob_var[x] += prob ** 2
                        d_route = res_route[x] - np.reshape(nextstate_dic[x], (1, nextstate_len))
                        close_route_res[x] += np.abs(d_route[action_dic[x]])
                        close_route_var[x] += d_route[action_dic[x]] ** 2
                        min_route = np.argmin(np.sum(d_route ** 2, axis=1))
                        maximum_route_res[x] += np.abs(d_route[min_route])
                        maximum_route_var[x] += d_route[min_route] ** 2
                    res_num += 1
            
        for x in range(agent_count):
            for t in range(6):
                log_txt.write(str(param_vectors[x][t]) + "\t")
            log_txt.write(str(prob_res[x] / res_num) + "\t")
            log_txt.write(str(prob_var[x] / res_num - (prob_res[x] / res_num) ** 2))
            for t in range(6):
                log_txt.write("\t" + str(close_route_res[x][t] / res_num))
            for t in range(6):
                log_txt.write("\t" + str(close_route_var[x][t] / res_num - (close_route_res[x][t] / res_num) ** 2))
            for t in range(6):
                log_txt.write("\t" + str(maximum_route_res[x][t] / res_num))
            for t in range(6):
                log_txt.write("\t" + str(maximum_route_var[x][t] / res_num - (maximum_route_res[x][t] / res_num) ** 2))
            log_txt.write("\n")
            