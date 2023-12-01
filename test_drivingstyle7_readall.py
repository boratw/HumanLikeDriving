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
from network.DrivingStyle11_bayesian_latent import DrivingStyleLearner
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
    d_state = [[state_vectors[step_start_index + step][i][0], state_vectors[step_start_index + step][i][1], state_vectors[step_start_index + step][i][2] * 0.017453293] for  i in range(agent_count)]
    res = json.dumps({"state" : d_state})

    global current_step
    current_step = step
    return res

def SendExpInfo(nothing):
    res = "{\"max_step\":" + str(step_count) + \
        ", \"agent_count\":" + str(agent_count) + \
        ", \"latent_len\":" + str(global_latent_len) + "}"
    return res

def SendLatents(list):
    target = int(list[0])
    o_mu = global_latent_mu[target]
    res = json.dumps({"mu" : o_mu}, cls=MyEncoder)
    return res

def SendPredLatent(list):
    target = int(list[0])
    start = int(list[1])
    end = int(list[2])
    if start > end:
        start = end
    m = np.zeros((global_latent_len, ))
    for x in range(start, end + 1):
        m += global_latent_mu[target][x]
    if m.any() != 0:
        d = np.sqrt(np.sum(m ** 2))
        v = np.exp(-d)
        m /= d
    else:
        v = 1.
    res = {"mu" : m, "std" : [v] * action_len}
    return res
    

def SendOutput(list):
    target = int(list[0])
    latent = SendPredLatent([target, current_step, current_step])
    d_state = [[state_vectors[step_start_index + current_step + j][target][0], state_vectors[step_start_index + current_step + j][target][1]] for j in range(0, 60, 15)]
    l_state = [ [[0., 0., 0., 0., 0., 0. ]] for j in range(action_len)]
    o_action = [0.] * (action_len + 1)
    global_latent_dic = [latent["mu"] for _ in range(16) ]
    
    traced, tracec = lane_tracers[target].Trace(state_vectors[step_start_index + current_step][target][0], state_vectors[step_start_index + current_step][target][1])

    state_dic = [cur_history[target][current_step][0] for _ in range(16)]
    route_dic = [cur_history[target][current_step][2] for _ in range(16)]
    with sess.as_default():
        res_route_mean, res_route_var, res_action = learner.get_output(state_dic, route_dic, global_latent_dic)
    res_route_std = np.sqrt(res_route_var)
    o_action[0] = np.mean(res_action[:, 0])
    o_action[1] = np.mean(res_action[:, 1])
    o_action[2] = np.mean(res_action[:, 2])
    o_action[3] = np.std(res_action)
    
    if tracec[1] == False :
        o_action[0] += o_action[1]
        o_action[1] = 0.
    if tracec[2] == False :
        o_action[0] += o_action[2]
        o_action[2] = 0.

    i = np.argmax([o_action[0], o_action[1], o_action[2]])
    
    o_mean =  np.mean(res_route_mean[:, i, :], axis=0)
    o_ep_var = np.mean(np.sqrt(res_route_std[:, i, :]), axis=0)
    o_al_var = np.std(res_route_mean[:, i, :], axis=0)
    x = 0.
    y = 0.
    l_state = []
    for j in range(3):
        x += o_mean[j * 2]
        y += o_mean[j * 2 + 1]
        l_state.append([float(x), float(y), float(np.sqrt(o_ep_var[j * 2])),
                            float(np.sqrt(o_ep_var[j * 2 + 1])), float(np.sqrt(o_al_var[j * 2])), float(np.sqrt(o_al_var[j * 2 + 1]))])
        
    res = {"route" : d_state, "predicted" : l_state, "action_prob" : o_action}
    print("vehicle " + str(target) + " step " + str(current_step))
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
route_len = 16
action_len = 3
global_latent_len = 4
num_of_agents = 4

pkl_index = 18
exp_index = 1

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, route_len=route_len, action_len= action_len, istraining=False)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    learner_saver.restore(sess, "train_log/DrivingStyle11_Bayesian_Latent/log_2023-09-19-18-22-43_100.ckpt")

    with open("data/gathered_from_npc_batjeon6/data_" + str(pkl_index) + ".pkl","rb") as fr:
        data = pickle.load(fr)


    state_vectors = data[exp_index]["state_vectors"]
    control_vectors = data[exp_index]["control_vectors"]
    agent_count = len(state_vectors[0])
    lane_tracers = [LaneTrace(laneinfo, 8) for _ in range(agent_count)]
    cur_history = [[] for _ in range(agent_count)]
    torque_added = [0 for _ in range(agent_count)]

    step_start_index = 200
    step_count = len(state_vectors) - step_start_index - 150

    for step in range(step_start_index, step_start_index+step_count):
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
                    
            trace_result = 0
            mindist = 99999
            for j, trace, c in zip(range(action_len), traced, tracec):
                if c:
                    dist = (trace[7][0] - state_vectors[step + 45][i][0]) ** 2 + (trace[7][1] - state_vectors[step + 45][i][1]) ** 2
                    if dist < mindist:
                        trace_result = j
                        mindist = dist



            cur_history[i].append( [np.concatenate([[velocity, (1. if state_vectors[step][5] == 0. else 0.), px, py, control_vectors[step][i][1]], np.array(other_vcs).flatten()]), nextstate, route, trace_result, torque_added[i]])
            if torque_added[i] > 0:
                torque_added[i] -= 1

    global_latent_mu = [[] for _ in range(agent_count)]

    for step in range(0, step_count):
        if step % 100 == 0:
            print("Getting latent from step " + str(step))
        state_dic = []
        nextstate_dic = []
        route_dic = []
        action_dic = []
        for x in range(agent_count):
            state_dic.append(cur_history[x][step][0])
            nextstate_dic.append(cur_history[x][step][1])
            route_dic.append(cur_history[x][step][2])
            action_dic.append(cur_history[x][step][3])

        res_mu = learner.get_latent(state_dic, nextstate_dic, route_dic, action_dic)
        for x in range(agent_count):
            if cur_history[x][step][4] == 0:
                global_latent_mu[x].append(res_mu[x])
            else:
                global_latent_mu[x].append(np.zeros((global_latent_len, )))

    for x in range(agent_count):
        with open("readed_" + str(x) + ".tsv", "wt") as f:
            f.write("""pos_x\tpos_y\t\
gt_1_x\tgt_1_y\tgt_2_x\tgt_2_y\tgt_3_x\tgt_3_y\t\
pred_1_x\tpred_1_y\tpred_2_x\tpred_2_y\tpred_3_x\tpred_3_y\t\
ep_1_x\tep_1_y\tep_2_x\tep_2_y\tep_3_x\tep_3_y\t\
al_1_x\tal_1_y\tal_2_x\tal_2_y\tal_3_x\tal_3_y
""")
            for current_step in range(0, step_count):
                px = state_vectors[current_step][x][0]
                py = state_vectors[current_step][x][1]
                yawsin = np.sin(state_vectors[current_step][x][2]  * -0.017453293)
                yawcos = np.cos(state_vectors[current_step][x][2]  * -0.017453293)

                rx1 = state_vectors[current_step + 15][x][0] - px
                ry1 = state_vectors[current_step + 15][x][1] - py
                px1, py1 = rotate(rx1, ry1, yawsin, yawcos)

                rx2 = state_vectors[current_step + 30][x][0] - px
                ry2 = state_vectors[current_step + 30][x][1] - py
                px2, py2 = rotate(rx2, ry2, yawsin, yawcos)

                rx3 = state_vectors[current_step + 45][x][0] - px
                ry3 = state_vectors[current_step + 45][x][1] - py
                px3, py3 = rotate(rx3, ry3, yawsin, yawcos)
                res=SendOutput([x])
                f.write(f"""{px}\t{py}\t\
{px1}\t{py1}\t{px2}\t{py2}\t{px3}\t{py3}\t\
{res["predicted"][0][0]}\t{res["predicted"][0][1]}\t{res["predicted"][1][0]}\t{res["predicted"][1][1]}\t{res["predicted"][2][0]}\t{res["predicted"][2][1]}\t\
{res["predicted"][0][2]}\t{res["predicted"][0][3]}\t{res["predicted"][1][2]}\t{res["predicted"][1][3]}\t{res["predicted"][2][2]}\t{res["predicted"][2][3]}\t\
{res["predicted"][0][4]}\t{res["predicted"][0][5]}\t{res["predicted"][1][4]}\t{res["predicted"][1][5]}\t{res["predicted"][2][4]}\t{res["predicted"][2][5]}
""")



