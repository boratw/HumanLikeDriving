
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
from network.DrivingStyle4 import DrivingStyleLearner
from network.DrivingStyleControl import DrivingStyleController
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import multiprocessing

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")

state_len = 152
nextstate_len = 10
agent_for_each_train = 8
global_latent_len = 4
local_latent_len = 1
l2_regularizer_weight = 0.0001

action_len = 2,


log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/DrivingStyle4/log_" + log_name + ".txt", "wt")

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

def parallel_task(item):
    x = item[0].location.x
    y = item[0].location.y

    traced = lane_tracers[i].Trace(x, y)
    if traced != None:

    else
        return None


vehicles_list = []
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

tf.disable_eager_execution()
sess = tf.Session()
try:
    world = client.get_world()

    settings = world.get_settings()
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprints = [x for x in blueprints if 
                    x.id.endswith('a2') or
                    x.id.endswith('etron') or
                    x.id.endswith('tt') or
                    x.id.endswith('grandtourer') or
                    x.id.endswith('impala') or
                    x.id.endswith('c3') or
                    x.id.endswith('charger_2020') or
                    x.id.endswith('crown') or
                    x.id.endswith('mkz_2017') or
                    x.id.endswith('mkz_2020') or
                    x.id.endswith('coupe') or
                    x.id.endswith('coupe_2020') or
                    x.id.endswith('cooper_s') or
                    x.id.endswith('cooper_s_2021') or
                    x.id.endswith('mustang') or
                    x.id.endswith('micra') or
                    x.id.endswith('leon') or
                    x.id.endswith('model3') or
                    x.id.endswith('prius')]

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)


    with sess.as_default():
        with multiprocessing.Pool(processes=50) as pool:
            learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, agent_for_each_train=agent_for_each_train, global_latent_len=global_latent_len, 
                                            local_latent_len= local_latent_len, l2_regularizer_weight=l2_regularizer_weight)
            learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
            controller = DrivingStyleController(state_len=state_len, action_len=action_len, global_latent_len=global_latent_len, l2_regularizer_weight=l2_regularizer_weight)
            controller_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
            sess.run(tf.global_variables_initializer())
            learner_saver.restore(sess, "train_log/DrivingStyle4/log_12-05-2023-15-34-16_2250.ckpt")

            controller.network_initialize()
            log_file.write("Epoch" + controller.log_caption() + "\n")

            history = []
            lane_tracers = [LaneTrace(laneinfo) for _ in range(50)]

            for epoch in range(1, 10000):
                random.shuffle(spawn_points)
                i = 0
                while len(vehicles_list) < 50:
                    blueprint = random.choice(blueprints)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)
                        blueprint.set_attribute('color', color)
                    actor = world.try_spawn_actor(blueprint, spawn_points[i])
                    if actor:
                        vehicles_list.append(actor)
                
                for step in range(1000):
                    states = []
                    state_vectors = []
                    for actor in vehicles_list:
                        tr = actor.get_transform()
                        v = actor.get_velocity()
                        try:
                            tl = actor.get_traffic_light_state()
                        except:
                            tl = carla.TrafficLightState.Unknown
                        states.append([tr, v, tl])
                    
                    for result in pool.imap(parallel_task, states):
                        state_vectors.append(result)



                pkl_index = random.randrange(51)
                with open("data/gathered_from_npc_batjeon2/data_" + str(pkl_index) + ".pkl","rb") as fr:
                    data = pickle.load(fr)
                print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

                history_data = []
                for result in pool.imap_unordered(parallel_task, data):
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
                    learner_saver.save(sess, "train_log/DrivingStyle4/log_" + log_name + "_" + str(epoch) + ".ckpt")


finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)