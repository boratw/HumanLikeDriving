
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
from network.DrvStylePredictor import DriveStylePredictor
from datetime import datetime
import numpy as np
import pickle
import time
import random

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")

routetracers = [RouteTracer(laneinfo) for _ in range(50)]
history = []


log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/DrvStylePredictor/log_" + log_name + ".txt", "wt")

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    learner = DriveStylePredictor()
    learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
    sess.run(tf.global_variables_initializer())
    learner.network_initialize()
    log_file.write("Epoch\tIteration" + learner.log_caption() + "\n")


    for epoch in range(1, 10000):
        pkl_index = random.randrange(1, 65)
        with open("data/gathered_from_npc/data_" + str(pkl_index) + ".pkl","rb") as fr:
            data = pickle.load(fr)
        print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

        for iteration in range(100):
            while len(history) < 10000:
                exp_index = random.randrange(len(data))
                print("Load Data #" + str(exp_index))

                agent_count = len(data[exp_index]["state_vectors"][0])
                params = np.array(data[exp_index]["params"]) * \
                    np.array([1 / 0.8, 1 / 0.5, 1 / 0.05, 1 / 0.05, 1 / 2.0, 1 / 1.0, 1 / 0.6])
                for state_vector in data[exp_index]["state_vectors"]:
                    for i in range(agent_count):
                        traced = routetracers[i].Trace(state_vector[i][0], state_vector[i][1], state_vector[i][2])
                        if traced != None and random.random() < 0.1:
                            history.append([state_vector[i], traced, params[i]])

            for step in range(64):
                dic = random.sample(range(len(history)), 64)

                state_dic = [history[x][0] for x in dic]
                route_dic = [history[x][1] for x in dic]
                target_dic = [history[x][2] for x in dic]

                learner.optimize_batch(state_dic, route_dic, target_dic)

            learner.log_print()
            log_file.write(str(epoch) + "\t" + str(iteration) + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()

            history = history[(len(history) // 8):]

        if epoch % 50 == 0:
            learner_saver.save(sess, "train_log/DrvStylePredictor/log_" + log_name + "_" + str(epoch) + ".ckpt")

