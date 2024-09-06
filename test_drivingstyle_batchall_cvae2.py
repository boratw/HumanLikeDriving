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
from network.DrivingStyle_Latent_CVae import DrivingStyleLearner
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import itertools

module_n = int(sys.argv[1])

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")


def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos


state_len = 83
prevstate_len = 6
nextstate_len = 6
agent_num = 100
agent_count = 100
action_len = 31
param_len = 7
global_latent_len = 4
pred_num = 31


tf.disable_eager_execution()
sess = tf.Session()


log_txt = open("test_log/log4_4/module" + str(module_n) + ".txt", "wt")

with sess.as_default():
    learner = DrivingStyleLearner(state_len=state_len - 1, prevstate_len=prevstate_len, nextstate_len=nextstate_len, action_len=action_len, param_len=param_len,
                                route_loss_weight=[4.0, 4.0, 2.0, 2.0, 1.0, 1.0], action_loss_weight=0.1)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    learner_saver.restore(sess, "train_log/DrivingStyle_Latent_CVae3/log_2024-08-20-13-54-15_1760.ckpt")

    pkl_names = os.listdir("data/gathered_from_npc4_2")
    pkl_names.sort()
    for pkl_name in pkl_names:
        print("Read data " + pkl_name)
        with open("data/gathered_from_npc4_2/" + pkl_name,"rb") as fr:
            data = pickle.load(fr)

        prob_res = np.zeros(agent_count)
        prob_var = np.zeros(agent_count)
        close_route_res = np.zeros((agent_count, nextstate_len))
        maximum_route_res = np.zeros((agent_count, nextstate_len))
        close_route_var = np.zeros((agent_count, nextstate_len))
        maximum_route_var = np.zeros((agent_count, nextstate_len))
        global_latent = np.zeros((agent_count, global_latent_len))
        global_latent_mean = np.zeros((agent_count, global_latent_len))
        global_latent_div = np.zeros((agent_count, global_latent_len)) + 1.
        res_num = np.zeros(agent_count)

        step_prob_res = np.zeros(500)
        step_prob_var = np.zeros(500)
        step_close_route_res = np.zeros((500, nextstate_len))
        step_maximum_route_res = np.zeros((500, nextstate_len))
        step_close_route_var = np.zeros((500, nextstate_len))
        step_maximum_route_var = np.zeros((500, nextstate_len))

        for exp_index in range(len(data)):
            with open("test_log/log4_4/module" + str(module_n) + "_" + pkl_name + "_" + str(exp_index) + ".txt", "wt") as log_exp_steps :
                if module_n == 1:
                    global_latent = np.zeros((agent_count, global_latent_len))
                    global_latent_mean = np.zeros((agent_count, global_latent_len))
                    global_latent_div = np.zeros((agent_count, global_latent_len)) + 1.
                state_vectors = data[exp_index]["state_vectors"]
                param_vectors = data[exp_index]["params"]
                agent_count = len(state_vectors[0])
                lane_tracers = [LaneTrace(laneinfo, 5) for _ in range(agent_count)]
                cur_history = [[] for _ in range(agent_count)]

                step_start_index = 200
                step_count = len(state_vectors) - step_start_index - 150

                for step in range(step_start_index, step_start_index+step_count, 4):
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
                        tstate = 1. if (state_vectors[step][i][5] == carla.TrafficLightState.Red or 
                                        state_vectors[step][i][5] == carla.TrafficLightState.Yellow) \
                                    else 0.
                                

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
                        
                        cur_history[i].append( [np.concatenate([[velocity, tstate, px, py],
                                                                np.array(other_vcs).flatten(), np.array(route).flatten()]), 
                                                nextstate, action])


                if module_n == 2:
                    for step in range(step_count // 4):
                        if step % 100 == 0:
                            print("Getting latent from step " + str(step))
                        state_dic = []
                        nextstate_dic = []
                        for x in range(agent_count):
                            state_dic.append(cur_history[x][step][0])
                            nextstate_dic.append(cur_history[x][step][1])

                        res_mu = learner.get_latent(state_dic, nextstate_dic, discrete=True)
                        global_latent += res_mu * np.sum(res_mu ** 2, axis=1, keepdims=True)
                        global_latent_div += np.sum(res_mu ** 2, axis=1, keepdims=True)
                    global_latent_mean = global_latent / global_latent_div

                for step in range(step_count // 4):
                    if step % 100 == 0:
                        print("Parsing step " + str(step))
                    state_dic = []
                    nextstate_dic = []
                    action_dic = []
                    for x in range(agent_count):
                        state_dic.append(cur_history[x][step][0])
                        nextstate_dic.append(cur_history[x][step][1])
                        action_dic.append(cur_history[x][step][2])

                    res_mu = learner.get_latent(state_dic, nextstate_dic, discrete=True)
                    if module_n == 1:
                        global_latent += res_mu * np.sum(res_mu ** 2, axis=1, keepdims=True)
                        global_latent_div += np.sum(res_mu ** 2, axis=1, keepdims=True)
                        global_latent_mean = global_latent / global_latent_div

                    if module_n == 0:
                        res_route, res_action, _ = learner.get_output_latent(state_dic, global_latent_mean, discrete=True)
                    elif  module_n == 1:
                        res_route, res_action, _ = learner.get_output_latent(state_dic, global_latent_mean * 0.5 + res_mu * 0.5, discrete=True)
                    else:
                        res_route, res_action, _ = learner.get_output_latent(state_dic, global_latent_mean * 0.5 + res_mu * 0.5, discrete=True)
                                        
                    for x in range(agent_count):
                        prob = (res_action[x][action_dic[x] - 1] + res_action[x][action_dic[x]] + res_action[x][action_dic[x] + 1])
                        max_action = np.argmax(res_action[x])
                        d_route = res_route[x] - np.reshape(nextstate_dic[x], (1, nextstate_len))

                        prob_res[x] += prob
                        prob_var[x] += prob ** 2
                        close_route_res[x] += np.abs(d_route[max_action])
                        close_route_var[x] += d_route[max_action] ** 2
                        min_route = np.argmin(np.sum(d_route ** 2, axis=1))
                        maximum_route_res[x] += np.abs(d_route[min_route])
                        maximum_route_var[x] += d_route[min_route] ** 2
                        res_num[x] += 1

                        step_prob_res[step] += prob
                        step_prob_var[step] += prob ** 2
                        step_close_route_res[step] += np.abs(d_route[max_action])
                        step_close_route_var[step] += d_route[max_action] ** 2
                        step_maximum_route_res[step] += np.abs(d_route[min_route])
                        step_maximum_route_var[step] += d_route[min_route] ** 2

                        if x == exp_index:
                            log_exp_steps.write("\t".join([str(x) for x in global_latent_mean[x]]) + "\t")
                            log_exp_steps.write(str(prob)+ "\t")
                            log_exp_steps.write("\t".join([str(np.abs(x)) for x in d_route[max_action]]) + "\n")

                                        

        for x in range(agent_count):
            prob_res[x] /= res_num[x]
            prob_var[x] /= res_num[x]
            close_route_res[x] /= res_num[x]
            close_route_var[x] /= res_num[x]
            maximum_route_res[x] /= res_num[x]
            maximum_route_var[x] /= res_num[x]

        for x in range(500):
            step_prob_res[x] /= (agent_count * len(data))
            step_prob_var[x] /= (agent_count * len(data))
            step_close_route_res[x] /= (agent_count * len(data))
            step_close_route_var[x] /= (agent_count * len(data))
            step_maximum_route_res[x] /= (agent_count * len(data))
            step_maximum_route_var[x] /= (agent_count * len(data))

        log_txt.write(pkl_name + "\t")
        prob_res_sum = np.mean(prob_res)
        prob_var_sum = np.mean(prob_var) 
        close_route_res_sum = np.mean(close_route_res, axis=0) 
        close_route_var_sum = np.mean(close_route_var, axis=0) 
        maximum_route_res_sum = np.mean(maximum_route_res, axis=0) 
        maximum_route_var_sum = np.mean(maximum_route_var, axis=0) 

        log_txt.write(str(prob_res_sum) + "\t")
        log_txt.write(str(prob_var_sum - (prob_res_sum) ** 2))
        for t in range(6):
            log_txt.write("\t" + str(close_route_res_sum[t]))
        for t in range(6):
            log_txt.write("\t" + str(close_route_var_sum[t] - (close_route_res_sum[t]) ** 2))
        for t in range(6):
            log_txt.write("\t" + str(maximum_route_res_sum[t]))
        for t in range(6):
            log_txt.write("\t" + str(maximum_route_var_sum[t] - (maximum_route_res_sum[t]) ** 2))
        log_txt.write("\n")
            

        with open("test_log/log4_4/module" + str(module_n) + "_" + pkl_name + ".txt", "wt") as log_exp_txt :
            for x in range(agent_count):
                for t in range(6):
                    log_exp_txt.write(str(param_vectors[x][t]) + "\t")
                log_exp_txt.write(str(prob_res[x]) + "\t")
                log_exp_txt.write(str(prob_var[x] - (prob_res[x]) ** 2))
                for t in range(6):
                    log_exp_txt.write("\t" + str(close_route_res[x][t]))
                for t in range(6):
                    log_exp_txt.write("\t" + str(close_route_var[x][t] - (close_route_res[x][t]) ** 2))
                for t in range(6):
                    log_exp_txt.write("\t" + str(maximum_route_res[x][t]))
                for t in range(6):
                    log_exp_txt.write("\t" + str(maximum_route_var[x][t] - (maximum_route_res[x][t]) ** 2))
                log_exp_txt.write("\n")
                

        with open("test_log/log4_4/module" + str(module_n) + "_" + pkl_name + "_step.txt", "wt") as log_exp_txt :
            for x in range(500):
                log_exp_txt.write(str(step_prob_res[x]) + "\t")
                log_exp_txt.write(str(step_prob_var[x] - (step_prob_res[x]) ** 2))
                for t in range(6):
                    log_exp_txt.write("\t" + str(step_close_route_res[x][t]))
                for t in range(6):
                    log_exp_txt.write("\t" + str(step_close_route_var[x][t] - (step_close_route_res[x][t]) ** 2))
                for t in range(6):
                    log_exp_txt.write("\t" + str(step_maximum_route_res[x][t]))
                for t in range(6):
                    log_exp_txt.write("\t" + str(step_maximum_route_var[x][t] - (step_maximum_route_res[x][t]) ** 2))
                log_exp_txt.write("\n")