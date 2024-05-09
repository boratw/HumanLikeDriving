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
from algorithm.routepredictor_DriveStyle import RoutePredictor_DriveStyle
from algorithm.routepredictor_DriveStyle_Nolatent import RoutePredictor_DriveStyle_Nolatent
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
from visualizer.server import VisualizeServer
import json
from datetime import datetime

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
    d_state = [[state_vectors[step_start_index + step][i][0], state_vectors[step_start_index + step][i][1], state_vectors[step_start_index + step][i][2] * 0.017453293] for  i in range(agent_count)]
    res = json.dumps({"state" : d_state})

    global current_step
    current_step = step_start_index + step
    return res

def SendExpInfo(nothing):
    res = json.dumps({"max_step" : step_count,
                      "agent_count" : agent_count,
                      "latent_len" : 4,
                      "param_vector" : param_vectors}, cls=MyEncoder)
    return res

def SetExpInfo(list):
    pkl_index = list[0]
    exp_index = list[1]
    return ""

def SendLatents(list):
    target = int(list[0])
    predictor, use_latent = predictor_list[int(list[1])]
    if use_latent != None:
        o_mu = predictor.global_latent_mean[target]
    else:
        o_mu = np.zeros(4)
    res = json.dumps({"mu" : o_mu}, cls=MyEncoder)
    return res

def SendPredLatent(list):
    '''
    target = int(list[0])
    predictor, use_latent = predictor_list[int(list[1])]
    m = np.zeros((global_latent_len, ))
    start = int(list[2])
    end = int(list[3])
    if start >= end:
        end = start + 1
    for x in range(start, end):
        m += predictor.global_latent_mean[target][x]
    res = json.dumps({"mu" : m / (end - start)}, cls=MyEncoder)
    return res
    '''
    return SendLatents(list)

    

def SendOutput(list):
    target = int(list[0])
    d_state = [[state_vectors[current_step + j][target][0], state_vectors[current_step + j][target][1]] for j in range(0, 80, 20)]
    #l_state = [ [[0., 0., 0., 0., 0., 0. ]] for j in range(pred_num)]
    
    state_dic = [cur_history[target][current_step - step_start_index][0]]
    #action_dic = [i for i in range(pred_num)]

    if len(list) == 7:
        latent = np.array([float(list[2]), float(list[3]), float(list[4]), float(list[5])])
    else:
        latent = np.zeros(4)
    '''
    elif len(list) == 4:
        start = int(list[1])
        end = int(list[2])
        if start >= end:
            end = start + 1
        m = np.zeros((global_latent_len, ))
        for x in range(start, end):
            m += global_latent_mu[target][x]
        latent = m / (end - start)

    '''
    predictor, use_latent = predictor_list[int(list[1])]
    
    with sess.as_default():
        if use_latent == True:
            if latent is None:
                latent = predictor.learner.global_latent_mean[target]
            latent_dic = [latent]
            l_state, l_prob, l_mask = predictor.learner.get_output(state_dic, latent_dic, discrete=True)
        elif use_latent == False:
            l_state, l_prob, l_mask = predictor.learner.get_output(state_dic, np.zeros((1, 4)), discrete=True)
        else:
            l_state, l_prob = predictor.learner.get_output(state_dic, discrete=True)
            l_mask = np.zeros((1, state_len))

        
    l_prob = np.mean(l_prob, axis=0)
    res = json.dumps({"route" : d_state, "predicted" : l_state[0], "action_prob" : l_prob, "mask" : l_mask[0], "latent" : latent}, cls=MyEncoder)
    print("vehicle " + str(target) + " step " + str(current_step))
    return res

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos



laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

state_len = 83
prevstate_len = 6
nextstate_len = 6
agent_num = 100
action_len = 31
global_latent_len = 4
pred_num = 31

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    predictor_default = RoutePredictor_DriveStyle_Nolatent(laneinfo, npc_count=agent_num, sess=sess, name="_nolatent", 
                                                 snapshot="train_log/DrivingStyle_NoLatent2/log_2023-12-05-20-11-44_780.ckpt")
    predictor_latent = RoutePredictor_DriveStyle(laneinfo, npc_count=agent_num, sess=sess, name="_latent", 
                                                 snapshot="train_log/DrivingStyle_Latent4_nextstate2/log_2023-12-28-14-55-59_1640.ckpt")
    predictor_avglatent = RoutePredictor_DriveStyle(laneinfo, npc_count=agent_num, sess=sess, name="_avglatent", 
                                                 snapshot="train_log/DrivingStyle_AvgLatent2/2024-02-19-16-24-49_300.ckpt")
    
    predictor_list = [(predictor_default, None), (predictor_latent, False),(predictor_latent, True),
                      (predictor_avglatent, False),(predictor_avglatent, True)]
    pkl_index = 0
    exp_index = 5

    loaded_pkl_index = -1
    loaded_exp_index = -1

    server = VisualizeServer()
    server.handlers["curstate"] = SendCurstate
    server.handlers["expinfo"] = SendExpInfo
    server.handlers["predictlatent"] = SendPredLatent
    server.handlers["latents"] = SendLatents
    server.handlers["predictroute"] = SendOutput
    server.handlers["setexp"] = SetExpInfo

    while True:
        if loaded_pkl_index != pkl_index:
            print("Data Load start : " + "data_" + str(pkl_index) + ".pkl")
            with open("data/gathered_from_npc1/data_" + str(pkl_index) + ".pkl","rb") as fr:
                data = pickle.load(fr)
            loaded_pkl_index = pkl_index


        print("Exp Load start : " + str(exp_index))
        state_vectors = data[exp_index]["state_vectors"]
        control_vectors = data[exp_index]["control_vectors"]
        param_vectors = data[exp_index]["params"]
        agent_count = len(state_vectors[0])
        lane_tracers = [LaneTrace(laneinfo, 5) for _ in range(agent_count)]
        cur_history = [[] for _ in range(agent_count)]

        step_start_index = 50
        step_count = len(state_vectors) - step_start_index - 150


        for step in range(step_start_index, step_start_index+step_count):
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
                        

                cur_history[i].append( [np.concatenate([[velocity, (1. if state_vectors[step][i][5] == 0. else 0.), px, py, control_vectors[step][i][1]],
                                                        np.array(other_vcs).flatten(), np.array(route).flatten()]), 
                                        nextstate])

        global_latent_mu = [[] for _ in range(agent_count)]
        global_latent = []
        

        predictor_latent.Reset()
        predictor_avglatent.Reset()
        for step in range(step_count):
            if step % 100 == 0:
                print("Getting latent from step " + str(step))
            state_dic = []
            nextstate_dic = []
            for x in range(agent_count):
                state_dic.append(cur_history[x][step][0])
                nextstate_dic.append(cur_history[x][step][1])
            predictor_latent.Add_Global_Latent_by_State(state_dic, nextstate_dic)
            predictor_avglatent.Add_Global_Latent_by_State(state_dic, nextstate_dic)

        
        loaded_exp_index = exp_index
        try:
            while pkl_index == loaded_pkl_index and exp_index == loaded_exp_index:
                server.Receive()
        except:
            break
    server.Destroy()