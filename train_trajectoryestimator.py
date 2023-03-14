
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

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")

routetracers = [RouteTracer(laneinfo) for _ in range(50)]
history = []


log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/TrajectoryEstimator/log_" + log_name + ".txt", "wt")

def rotate(posx, posy, yaw):
    return posx * np.sin(yaw) + posy * np.cos(yaw), posx * np.cos(yaw) - posy * np.sin(yaw)

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = TrajectoryEstimator()
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    learner.network_initialize()
    log_file.write("Epoch\tIteration" + learner.log_caption() + "\n")


    for epoch in range(1, 10000):
        pkl_index = random.randrange(7)
        with open("data/gathered_from_npc2/data_" + str(pkl_index) + ".pkl","rb") as fr:
            data = pickle.load(fr)
        print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

        for iteration in range(20):
            while len(history) < 50000:
                exp_index = random.randrange(len(data))
                print("Load Data #" + str(exp_index))

                state_vectors = data[exp_index]["state_vectors"]
                agent_count = len(data[exp_index]["state_vectors"][0])
                for step, state_vector in enumerate(state_vectors[:-20]):
                    for i in range(agent_count):
                        if state_vector[i][6] == carla.VehicleFailureState.NONE:
                            if random.random() < 0.1:
                                traced = routetracers[i].Trace(state_vector[i][0], state_vector[i][1], state_vector[i][2])
                                if traced != None:
                                    other_vcs = []
                                    for j in range(agent_count):
                                        if i != j:
                                            relposx = state_vector[j][0] - state_vector[i][0]
                                            relposy = state_vector[j][1] - state_vector[i][1]
                                            px, py = rotate(relposx, relposy, state_vector[i][2]  * 0.017453293)
                                            vx, vy = rotate(state_vector[j][3], state_vector[j][4], state_vector[i][2]  * 0.017453293)
                                            relyaw = state_vector[j][2] - state_vector[i][2]
                                            if relyaw < -np.pi:
                                                relyaw += 2 * np.pi
                                            elif relyaw > -np.pi:
                                                relyaw -= 2 * np.pi
                                            other_vcs.append([relposx, relposy, relyaw, vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
                                    other_vcs = np.array(sorted(other_vcs, key=lambda s: s[5]))
                                    velocity = np.sqrt(state_vector[i][3] ** 2 + state_vector[i][4] ** 2)
                                    route = []
                                    for j in range(0, 20, 2):
                                        relposx = state_vectors[step+j][i][0] - state_vector[i][0]
                                        relposy = state_vectors[step+j][i][1] - state_vector[i][1]
                                        px, py = rotate(relposx, relposy, state_vector[i][2]  * 0.017453293)
                                        route.append([px, py])

                                    history.append( [[velocity, state_vector[i][5]], traced, other_vcs[:8, :5], route])


                        

            for step in range(100):
                dic = random.sample(range(len(history)), 128)

                state_dic = [history[x][0] for x in dic]
                route_dic = [history[x][1] for x in dic]
                othervcs_dic = [history[x][2] for x in dic]
                target_dic = [history[x][3] for x in dic]

                learner.optimize_batch(state_dic, route_dic, othervcs_dic, target_dic)

            learner.log_print()
            log_file.write(str(epoch) + "\t" + str(iteration) + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()

            history = history[(len(history) // 32):]

        if epoch % 50 == 0:
            learner_saver.save(sess, "train_log/TrajectoryEstimator/log_" + log_name + "_" + str(epoch) + ".ckpt")

