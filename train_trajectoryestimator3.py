
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
from network.TrajectoryEstimator3 import TrajectoryEstimator
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import multiprocessing

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")



log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/TrajectoryEstimator2/log_" + log_name + ".txt", "wt")

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
    history_exp = [[] for _ in range(50)]

    state_vectors = item["state_vectors"]
    agent_count = len(item["state_vectors"][0])

    stepstart = random.randrange(10)
    for step, state_vector in enumerate(state_vectors[stepstart:-60:10]):
        for i in range(agent_count):
            other_vcs = []
            x = state_vector[i][0]
            y = state_vector[i][1]
            yawsin = np.sin(state_vector[i][2]  * -0.017453293)
            yawcos = np.cos(state_vector[i][2]  * -0.017453293)
            for j in range(agent_count):
                if i != j:
                    relposx = state_vector[j][0] - x
                    relposy = state_vector[j][1] - y
                    px, py = rotate(relposx, relposy, yawsin, yawcos)
                    vx, vy = rotate(state_vector[j][3], state_vector[j][4], yawsin, yawcos)
                    relyaw = (state_vector[j][2] - state_vector[i][2])   * 0.017453293
                    if relyaw < -np.pi:
                        relyaw += 2 * np.pi
                    elif relyaw > np.pi:
                        relyaw -= 2 * np.pi
                    other_vcs.append([relposx, relposy, relyaw, vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
            other_vcs = np.array(sorted(other_vcs, key=lambda s: s[5]))
            velocity = np.sqrt(state_vector[i][3] ** 2 + state_vector[i][4] ** 2)
            route = []
            for j in range(10, 60, 10):
                relposx = state_vectors[step+j][i][0] - x
                relposy = state_vectors[step+j][i][1] - y
                px, py = rotate(relposx, relposy, yawsin, yawcos)
                route.append([px, py])
            waypoints = []
            option = [0., 0., 0.]
            px, py = 0., 0.
            prevx = 0.
            prevy = 0.
            k = step
            for j in range(3):
                while k < len(state_vectors):
                    if len(state_vectors[k][i][8]) > 0 :
                        if state_vectors[k][i][8][0][1] != prevx or state_vectors[k][i][8][0][2] != prevy:
                            relposx = state_vectors[k][i][8][0][1] - x
                            relposy = state_vectors[k][i][8][0][2] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            if state_vectors[k][i][8][0][0] in ReadOption:
                                option = ReadOption[state_vectors[k][i][8][0][0]]
                            else:
                                print("Unknown RoadOption " + state_vectors[k][i][8][0][0])
                            prevx = state_vectors[k][i][8][0][1]
                            prevy = state_vectors[k][i][8][0][2]
                            break
                    k += 1
                waypoints.append([option[0], option[1], option[2], px, py])
                
            px, py = 9999., 9999.
            for t in state_vector[i][6]:
                if np.sqrt(px * px + py * py) >  np.sqrt((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                    px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
            if px == 9999.:
                px = 0.
                py = 0.
            history_exp[i].append( [[velocity, state_vector[i][5], px, py], waypoints, other_vcs[:8, :5], route])
    return history_exp

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    with multiprocessing.Pool(processes=20) as pool:
        learner = TrajectoryEstimator()
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        sess.run(tf.global_variables_initializer())
        learner.network_initialize()
        log_file.write("Epoch" + learner.log_caption() + "\n")

        history = []

        for epoch in range(1, 10000):
            pkl_index = random.randrange(22)
            with open("data/gathered_from_param3_npc/data_" + str(pkl_index) + ".pkl","rb") as fr:
                data = pickle.load(fr)
            print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

            history_data = []
            for result in pool.imap(parallel_task, data):
                print("Load Data")
                history_data.append(result)
            history.append(history_data)

            print("Current History Length : " + str(len(history)))
            for iter in range(len(history)):
                print("Train Step #" + str(iter))

                min_step = [999999] * 4
                data_index = random.randrange(len(history))
                local_latents = []
                print("Local latent optimizing")
                for local_step in range(4):
                    exp_index = random.randrange(len(history[data_index]))
                    cur_history = history[data_index][exp_index]
                    agent_num = len(cur_history)

                    global_loss_sum = 0
                    local_loss_sum = 0

                    print("Read data " + str(data_index) + " exp " + str(exp_index))

                    step_dic = [ list(range(len(cur_history[x]))) for x in range(agent_num) ]
                    min_step[local_step] = 999999
                    for s in step_dic:
                        random.shuffle(s)
                        if len(s) < min_step[local_step]:
                            min_step[local_step] = len(s)
                    
                    local_latent = np.zeros([agent_num, min_step[local_step], 4])

                    for step in range(min_step[local_step]):
                        state_dic = []
                        waypoint_dic = []
                        othervcs_dic = []
                        route_dic = []
                        for x in range(agent_num):
                            state_dic.append(cur_history[x][step_dic[x][step]][0])
                            waypoint_dic.append(cur_history[x][step_dic[x][step]][1])
                            othervcs_dic.append(cur_history[x][step_dic[x][step]][2])
                            route_dic.append(cur_history[x][step_dic[x][step]][3] )

                        res, local_loss = learner.optimize_local(state_dic, waypoint_dic, othervcs_dic, route_dic)
                        local_loss_sum += local_loss
                        for x in range(agent_num):
                            local_latent[x][step_dic[x][step]] = res[x]
                    local_latents.append(local_latent)

                print("Global latent optimizing")
                for global_step in range(agent_num * 4):
                    agent_dic = random.choices(list(range(agent_num)), k=4)
                    latent_dic = []
                    for x in agent_dic:
                        for step in range(4):
                            start_step = random.randrange(min_step[step] - 100)
                            latent_dic.append(local_latents[step][x][start_step:(start_step + 100)])


                    global_loss = learner.optimize_global(latent_dic)
                    global_loss_sum += global_loss
                
            if len(history) > 32:
                history = history[1:]

            learner.log_print()
            log_file.write(str(epoch) + "\t" + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()


            if epoch % 50 == 0:
                learner_saver.save(sess, "train_log/TrajectoryEstimator3/log_" + log_name + "_" + str(epoch) + ".ckpt")

