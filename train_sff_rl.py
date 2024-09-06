import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('/home/user/carla-0.9.14/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import tensorflow.compat.v1 as tf

import numpy as np
from collections import deque
from algorithm.global_route_planner import GlobalRoutePlanner, RoadOption
from algorithm.controller import PIDLongitudinalController, PIDLateralController
from network.sac import SAC
from algorithm.safetypotential2 import SafetyPotential

from laneinfo import LaneInfo
from algorithm.actor import Actor

import random
import pickle
import math
from datetime import datetime

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos


log_name = "policy_train_log/Sff1/log_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = open(log_name + ".txt", "wt")

npc_vehicles_list = []
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

agent_num = 80
state_len = 61
action_len = 1
tf.disable_eager_execution()
sess = tf.Session()

try:
    with sess.as_default():

        learner = SAC(state_len,  action_len, policy_gamma=0.95)
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        sff = SafetyPotential()
        init = tf.global_variables_initializer()
        sess.run(init)
        learner.network_initialize()
        log_file.write("Epoch" + learner.log_caption() + "\n")

        world = client.get_world()

        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)

        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        #settings.no_rendering_mode = True
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

        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        hero_actors = [Actor(world, client, blueprints=blueprints) for i in range(8)]
        latcontrollers = [PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05) for i in range(8)]
        history = []
        for exp in range(1000):
            
            distance_to_leading_vehicle = [ np.random.uniform(-5.0, 5.0) + 10.  for i in range(agent_num) ]
            vehicle_lane_offset = [ np.random.uniform(-1.0, 1.0) for i in range(agent_num) ]
            vehicle_speed = [ np.random.uniform(-50.0, 50.0) for i in range(agent_num) ]
            steer_ratio = [ np.clip(np.random.normal(0., 0.125), -0.25, 0.25) + 1.0 for i in range(agent_num) ]
            accel_ratio = [ np.clip(np.random.normal(0., 0.125), -0.25, 0.25) + 1.0 for i in range(agent_num) ]
            brake_ratio = [ np.clip(np.random.normal(0., 0.125), -0.25, 0.25) + 1.0 for i in range(agent_num) ]
            impatient_lane_change = [ np.random.uniform(200., 1200.)  for i in range(agent_num) ]

            desired_velocity = [ 11.1111 * (1.0 - vehicle_speed[i] / 100.0)  for i in range(agent_num) ]
            
            exp_str = str(exp)
            for iteration in range(16):
                print("exp " + exp_str + " : " + str(iteration))
                random.shuffle(spawn_points)
                npc_vehicles_list = []
                batch = []
                state_vectors = []
                control_vectors = []

                for n, transform in enumerate(spawn_points):
                    if n >= agent_num:
                        break
                    blueprint = random.choice(blueprints)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)

                        blueprint.set_attribute('color', color)
                    if blueprint.has_attribute('driver_id'):
                        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                        blueprint.set_attribute('driver_id', driver_id)
                    blueprint.set_attribute('role_name', 'autopilot')

                    # spawn the cars and set their autopilot and light state all together
                    batch.append(SpawnActor(blueprint, transform)
                        .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

                for response in client.apply_batch_sync(batch, True):
                    if response.error:
                        print(response.error)
                    else:
                        npc_vehicles_list.append(response.actor_id)

                all_npc_actors = world.get_actors(npc_vehicles_list)

                for i, actor in enumerate(all_npc_actors):
                    traffic_manager.distance_to_leading_vehicle(actor, distance_to_leading_vehicle[i] )
                    traffic_manager.vehicle_lane_offset(actor, vehicle_lane_offset[i])
                    traffic_manager.vehicle_percentage_speed_difference(actor, vehicle_speed[i])
                    traffic_manager.ignore_lights_percentage(actor, 0)
                    traffic_manager.ignore_signs_percentage(actor, 0)
                    traffic_manager.ignore_vehicles_percentage(actor, 0)
                    
                impatiece = [ 0.0 for i in range(agent_num) ]

                for i, x in enumerate(hero_actors):
                    x.reset()
                    latcontrollers[i].reset()
                    
                

                world.tick()
                
                for step in range(100):
                    vehiclecontrols = []
                    for i, actor in enumerate(all_npc_actors):
                        vc = actor.get_control()
                        vc.steer = 0.
                        vc.brake = 1.
                        vc.throttle = 0.
                        vehiclecontrols.append(carla.command.ApplyVehicleControl(actor, vc))
                    for i, actor in enumerate(hero_actors):
                        res = actor.step([0., 1., 0.], return_control=True)
                        vehiclecontrols.append(carla.command.ApplyVehicleControl(actor.player, res["control"]))
                    client.apply_batch(vehiclecontrols)
                    world.tick()

                prev_state_vectors = None
                prev_action_vectors = None
                survive_vectors = [True] * 8
                prev_steers = [0.] * 8
                for step in range(1000):
                    state_vector = []
                    vehiclecontrols = []
                    for i, actor in enumerate(all_npc_actors):
                        tr = actor.get_transform()
                        v = actor.get_velocity()

                        vel = np.sqrt(v.x * v.x + v.y * v.y)
                        if vel > 0.1:
                            if desired_velocity[i] > vel + 3.0:
                                impatiece[i] += (desired_velocity[i] * 1.5 - vel) * 0.1
                            else:
                                impatiece[i] = 0.
                        if impatiece[i] < 0.:
                            impatiece[i] = 0.
                        
                        if impatient_lane_change[i] < impatiece[i]:
                            traffic_manager.random_left_lanechange_percentage(actor, 100)
                            traffic_manager.random_right_lanechange_percentage(actor, 100)
                        else:
                            traffic_manager.random_left_lanechange_percentage(actor, 0)
                            traffic_manager.random_right_lanechange_percentage(actor, 0)

                        vc = actor.get_control()
                        vc.steer = np.clip(vc.steer * steer_ratio[i], -1.0, 1.0)
                        vc.brake = np.clip(vc.brake * brake_ratio[i], 0.0, 1.0)
                        vc.throttle = np.clip(vc.throttle * accel_ratio[i], 0.0, 1.0)


                        vehiclecontrols.append(carla.command.ApplyVehicleControl(actor, vc))

                        state = [tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y]
                        state_vector.append(state)
                        
                    for i, actor in enumerate(hero_actors):
                        tr = actor.player.get_transform()
                        v = actor.player.get_velocity()

                        state = [tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y]
                        state_vector.append(state)

                            
                    rl_states = []
                    for i, actor in enumerate(hero_actors):
                        tr = actor.player.get_transform()
                        v = actor.player.get_velocity()
                        x = tr.location.x
                        y = tr.location.y
                        yawsin = np.sin(tr.rotation.yaw * -0.017453293)
                        yawcos = np.cos(tr.rotation.yaw * -0.017453293)

                        distance_array = [(j[0] - x) ** 2 + (j[1] - y) ** 2 for j in state_vector]
                        distance_indicies = np.array(distance_array).argsort()

                        velocity = math.sqrt(v.x ** 2 + v.y ** 2)

                        other_vcs = []
                        for j in distance_indicies[1:9]:
                            relposx = state_vector[j][0] - x
                            relposy = state_vector[j][1] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            vx, vy = rotate(state_vector[j][3], state_vector[j][4], yawsin, yawcos)
                            relyaw = (state_vector[j][2] - state_vector[i][2])   * 0.017453293
                            other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy])
                        
                        v_prod = sff.get_potential(8.3333, prev_steers[i], tr, v, other_vcs)
                        rl_states.append(np.concatenate([[velocity], np.array(other_vcs).flatten(), v_prod]))
                    
                    rl_actions = learner.get_action(rl_states)
                    steers = []
                    for i, actor in enumerate(hero_actors):
                        tr = actor.player.get_transform()
                        steer = latcontrollers[i].run_step(actor.route[2][0].transform, tr)
                        steers.append(steer)
                        if rl_actions[i] < 0:
                            brake = float(-rl_actions[i])
                            accel = 0
                        else:
                            brake = 0
                            accel = float(rl_actions[i])
                        
                        if survive_vectors[i]:
                            res = actor.step([accel, brake, steer], return_control=True)
                            if(res["collision"]):
                                survive_vectors[i] = False
                                panelty = -10.
                                print("COLLISION")
                            elif res["velocity"] > 16:
                                survive_vectors[i] = False
                                panelty = -10.
                                print("HIGH SPEED")
                            else:
                                panelty = 0.
                            
                            vel_score = abs(res["velocity"] - 8.33333) * 0.01

                            if prev_state_vectors is not None:
                                history.append([prev_state_vectors[i], rl_states[i], prev_action_vectors[i], [vel_score + panelty], [survive_vectors[i]]])
                        else:
                            res = actor.step([0., 1., 0.], return_control=True)
                        vehiclecontrols.append(carla.command.ApplyVehicleControl(actor.player, res["control"]))

                    prev_state_vectors = rl_states
                    prev_action_vectors = rl_actions
                    prev_steers = steers

                    client.apply_batch(vehiclecontrols)
                    world.tick()

                client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicles_list])
                for x in hero_actors:
                    x.destroy()


            for iter in range(32):
                for iter2 in range(32):
                    dic = random.sample(range(len(history)), 64)

                    state_dic = [history[x][0] for x in dic]
                    nextstate_dic = [history[x][1] for x in dic]
                    action_dic = [history[x][2] for x in dic]
                    reward_dic = [history[x][3] for x in dic]
                    survive_dic = [history[x][4] for x in dic]

                    learner.optimize(state_dic, nextstate_dic, action_dic, reward_dic, survive_dic, exp)
                learner.network_intermediate_update()

            learner.log_print()
            log_file.write(str(exp) + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()

            if exp % 20 == 0:
                learner_saver.save(sess, log_name + "_" + str(exp) + ".ckpt")

            history = history[(len(history) // 32):]




finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(npc_vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicles_list])
    for x in hero_actors:
        x.destroy()


    time.sleep(1.)