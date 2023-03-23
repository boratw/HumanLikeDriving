
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

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")



log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/TrajectoryEstimator2/log_" + log_name + ".txt", "wt")

def rotate(posx, posy):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

ReadOption = { "LaneFollow" : [1., 0., 0.],
              "Left" : [0., 0., 1.],
              "Right" : [0., 0., -1.],
              "ChangeLaneLeft" : [0., 1., 0.],
              "ChangeLaneRight" : [0., -1, 0.],
              "Straight" : [1., 0., 0.]
              }


tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = TrajectoryEstimator(traj_len=10)
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    learner.network_initialize()
    log_file.write("Epoch" + learner.log_caption() + "\n")

    history = []

    for epoch in range(1, 10000):
        pkl_index = random.randrange(51)
        with open("data/gathered_from_param2_npc/data_" + str(pkl_index) + ".pkl","rb") as fr:
            data = pickle.load(fr)
        print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

        for exp_index in range(len(data)):
            print("Load Data #" + str(exp_index))
            history_exp = [[] for _ in range(50)]

            state_vectors = data[exp_index]["state_vectors"]
            agent_count = len(data[exp_index]["state_vectors"][0])
            for step, state_vector in enumerate(state_vectors[:-120]):
                for i in range(agent_count):
                    if len(state_vector[i][7]) > 0:
                        if random.random() < 0.1:
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
                            for j in range(10, 110, 10):
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
                                
                            if np.sqrt(px * px + py * py) > 1. or random.random() < 0.01:
                                history_exp[i].append( [[velocity, state_vector[i][5]], waypoints, other_vcs[:8, :5], route])
            history_exp = [ h for h in history_exp if len(h) > 10]
            history.append([99999999, history_exp])

        print("Current History Length : " + str(len(history)))
        for iter in range(len(history) // 8):
            exp_index = random.randrange(len(history))
            cur_history = history[exp_index][1]
            global_target = [[] for _ in range(len(cur_history))]
            for step in range(64):
                agent_dic = random.choices(list(range(len(cur_history))), k=32)
                step_dic = [ random.randrange(len(cur_history[x])) for x in agent_dic ]

                state_dic = []
                waypoint_dic = []
                othervcs_dic = []
                route_dic = []
                for x in range(32):
                    state_dic.append(cur_history[agent_dic[x]][step_dic[x]][0])
                    waypoint_dic.append(cur_history[agent_dic[x]][step_dic[x]][1])
                    othervcs_dic.append(cur_history[agent_dic[x]][step_dic[x]][2])
                    route_dic.append(cur_history[agent_dic[x]][step_dic[x]][3] )

                res, local_loss = learner.optimize_local(state_dic, waypoint_dic, othervcs_dic, route_dic)
                for x in range(32):
                    global_target[agent_dic[x]].append(res[x])

            global_target_exist = list(range(len(cur_history)))
            for x in range(len(cur_history)):
                if len(global_target[x]) > 10:
                    global_target[x] = np.mean(global_target[x], axis=0)
                else:
                    global_target[x] = None
                    global_target_exist.remove(x)
            for step in range(64):
                if len(global_target_exist) > 32:
                    agent_dic = random.sample(global_target_exist, k=32)
                else:
                    agent_dic = global_target_exist
                step_dic = [ random.randrange(len(cur_history[x])) for x in agent_dic ]

                state_dic = []
                waypoint_dic = []
                othervcs_dic = []
                route_dic = []
                target_dic = []
                for x in range(32):
                    state_dic.append(cur_history[agent_dic[x]][step_dic[x]][0])
                    waypoint_dic.append(cur_history[agent_dic[x]][step_dic[x]][1])
                    othervcs_dic.append(cur_history[agent_dic[x]][step_dic[x]][2])
                    route_dic.append(cur_history[agent_dic[x]][step_dic[x]][3] )
                    target_dic.append(global_target[agent_dic[x]] )

                global_loss = learner.optimize_global(state_dic, waypoint_dic, othervcs_dic, route_dic, target_dic)
            
            history[exp_index][0] = local_loss + global_loss
            print("Train Step #" + str(iter))

        history.sort(key=lambda s: s[0])
        history = history[len(history) // 16:]

        learner.log_print()
        log_file.write(str(epoch) + "\t" + learner.current_log() + "\n")
        log_file.flush()
        learner.network_update()


        if epoch % 50 == 0:
            learner_saver.save(sess, "train_log/TrajectoryEstimator2/log_" + log_name + "_" + str(epoch) + ".ckpt")

