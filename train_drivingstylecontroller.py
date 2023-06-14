
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
from network.DrivingStyle7_2 import DrivingStyleLearner
from network.DrivingControl import DrivingController
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import multiprocessing

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")

state_len = 53 
nextstate_len = 2
route_len = 6
action_len = 3
agent_for_each_train = 8
global_latent_len = 4
l2_regularizer_weight = 0.0001
global_regularizer_weight = 0.01

control_state_len = 4
control_action_len = 2
agent_num = 50

log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/DrivingControl/log_" + log_name + ".txt", "wt")

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos


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
        learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, agent_for_each_train=agent_for_each_train, global_latent_len=global_latent_len, 
                                    l2_regularizer_weight=l2_regularizer_weight, global_regularizer_weight=global_regularizer_weight, route_len=route_len, action_len= action_len)
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        controller = DrivingController(state_len=control_state_len, action_len=control_action_len, l2_regularizer_weight=l2_regularizer_weight)
        controller_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        sess.run(tf.global_variables_initializer())
        learner_saver.restore(sess, "train_log/DrivingStyle7/log_08-06-2023-18-48-38_150.ckpt")

        controller.network_initialize()
        log_file.write("Epoch" + controller.log_caption() + "\n")

        history = []
        lane_tracers = [LaneTrace(laneinfo, 3) for _ in range(agent_num)]
        impatiece = [ 0.0 for i in range(agent_num) ]
        control_traget = [ 0 for i in range(agent_num) ]
        prev_laneid = [ 0 for i in range(agent_num) ]
        torque_added = [ 0 for _ in range(agent_num) ]
        throttle_add = [ 0. for _ in range(agent_num) ]
        steer_add = [ 0. for _ in range(agent_num) ]

        for epoch in range(1, 10000):
            random.shuffle(spawn_points)
            i = 0
            while len(vehicles_list) < agent_num:
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                actor = world.try_spawn_actor(blueprint, spawn_points[i])
                if actor:
                    vehicles_list.append(actor)
                i += 1
            
            world.tick()
            world.tick()
            world.tick()
            world.tick()
            world.tick()
            latent = np.random.normal(0.0, 2.0, (agent_num, global_latent_len))
            result_vectors = []
            for step in range(1000):
                states = []
                state_vectors = []
                route_vectors = []
                control_state_vectors = []
                failed = []
                reward_mean = 0.
                for i, actor in enumerate(vehicles_list):
                    tr = actor.get_transform()
                    v = actor.get_velocity()
                    a = actor.get_acceleration()
                    forward = tr.get_forward_vector()
                    try:
                        tlight = actor.get_traffic_light()
                        tlight_state = tlight.get_state()
                        tlight_wps = tlight.get_stop_waypoints()
                        tlight_pos = [[w.transform.location.x, w.transform.location.y] for w in tlight_wps ]
                    except:
                        tlight_state = carla.TrafficLightState.Unknown
                        tlight_pos = []
                    states.append([tr.location.x, tr.location.x, tr.rotation.yaw, v.x, v.y, tlight_state, tlight_pos])
                    control_state_vectors.append([forward.dot(v), forward.dot(a)])
            
                for i, actor in enumerate(vehicles_list):
                    x = states[i][0]
                    y = states[i][1]
                    other_vcs = []

                    yawsin = np.sin(states[i][2] * -0.017453293)
                    yawcos = np.cos(states[i][2] * -0.017453293)
                    for j in range(agent_num):
                        if i != j:
                            relposx = states[j][1] - x
                            relposy = states[j][2] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            vx, vy = rotate(states[j][3], states[j][4], yawsin, yawcos)
                            relyaw = (states[j][2] - states[i][2])  * 0.017453293
                            other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
                    other_vcs = np.array(sorted(other_vcs, key=lambda s: s[6]))
                    velocity = np.sqrt(states[j][3] ** 2 +  states[j][4] ** 2)

                    if velocity > 0.1:
                        impatiece[i] += (30.0 - velocity - 5.0) * 0.01
                    else:
                        impatiece[i] = 0.
                    if impatiece[i] < 0.:
                        impatiece[i] = 0.

                    traced, tracec = lane_tracers[i].Trace(x, y)
                    failed.append(traced == None)
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
                    for t in states[i][6]:
                        if (px * px + py * py) >  ((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                            px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)

                    state_vectors.append(np.concatenate([[velocity, (1. if states[i][5] == 0 else 0.), px, py, impatiece[i]], other_vcs[:8,:6].flatten()]))
                    route_vectors.append(route)

                nextstate, action = learner.get_global_decoded(state_vectors, route_vectors, latent)
                for i, actor in enumerate(vehicles_list):
                    if action[i] != 0:
                        control_traget[i] = action[i]
                    control_state_vectors[i].extend(nextstate[i])

                control_action = controller.get_action(control_state_vectors)
                vehiclecontrols = []
                result_vector = []


                control_action = np.clip(control_action, -1, 1)
                for i, actor in enumerate(vehicles_list):
                    if random.random() < 0.05 * np.exp(-epoch / 100):
                        torque_added[i] = random.randrange(5, 10)
                        steer_add[i] = np.random.normal(0.0, 0.1)
                        throttle_add[i] = np.random.normal(0.25, 0.25)
                    if torque_added[i] >= 1:
                        if control_action[i][0] < 0.:
                            control_action[i][0] *= 0.25
                        control_action[i][0] += throttle_add[i]
                        control_action[i][1] += steer_add[i]
                        torque_added[i] -= 1

                control_action = np.clip(control_action, -1, 1)
                for i, actor in enumerate(vehicles_list):
                    vc = carla.VehicleControl()
                    if control_action[i][0] < 0.:
                        vc.throttle = 0.
                        vc.brake = -float(control_action[i][0]) * 0.5
                    else:
                        vc.throttle = float(control_action[i][0])
                        vc.brake = 0.
                    vc.steer = float(control_action[i][1])

                    result_vector.append([control_state_vectors[i], control_action[i], states[i][1], states[i][2], states[i][3]])
                    vehiclecontrols.append(carla.command.ApplyVehicleControl(actor, vc))
                
                client.apply_batch(vehiclecontrols)
                world.tick()
                result_vectors.append(result_vector)
                if len(result_vectors) >= 20:
                    for i, actor in enumerate(vehicles_list):
                        if failed[i] == False:
                            yawsin = np.sin(result_vectors[0][i][4] * -0.017453293)
                            yawcos = np.cos(result_vectors[0][i][4] * -0.017453293)
                            px, py = rotate(states[i][1] - result_vectors[0][i][2], states[i][2] - result_vectors[0][i][3], yawsin, yawcos)

                            reward = 1.5 - np.sqrt((px - result_vectors[0][i][0][2]) ** 2 + (py - result_vectors[0][i][0][3]) ** 2)
                            reward_mean += reward
                            if reward < 0.:
                                reward = 0.
                            history.append([result_vectors[0][i][0], result_vectors[0][i][1], result_vectors[1][i][0], [reward]])

                    result_vectors = result_vectors[1:]
                print("Step " + str(step) + " Reward " + str(reward_mean / agent_num))

            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list = []
            world.tick()

            print("Current History Length : " + str(len(history)))
            for iter in range(256):

                dic = random.sample(range(len(history)), len(history) if len(history) < 128 else 128)

                state_dic = [history[x][0] for x in dic]
                action_dic = [history[x][1] for x in dic]
                nextstate_dic = [history[x][2] for x in dic]
                reward_dic = [history[x][3] for x in dic]

                data_index = random.randrange(len(history))
                exp_index = random.randrange(len(history[data_index]))

                controller.optimize(epoch, state_dic, nextstate_dic, action_dic, reward_dic)
        
                
            history = history[(len(history) // 32 ):]

            controller.log_print()
            log_file.write(str(epoch) + "\t" + controller.current_log() + "\n")
            log_file.flush()
            controller.network_update()


            if epoch % 50 == 0:
                controller_saver.save(sess, "train_log/DrivingControl/log_" + log_name + "_" + str(epoch) + ".ckpt")


finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)