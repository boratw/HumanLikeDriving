import numpy as np
import cv2
import random
import tensorflow.compat.v1 as tf
import sys
import os
import glob
import time
import math
try:
    sys.path.append(glob.glob('/home/user/carla-0.9.14/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('../')
    sys.path.append('../algorithm/')
except IndexError:
    pass

import carla
import pickle
from laneinfo import LaneInfo
from algorithm.controller import PIDLongitudinalController, PIDLateralController
from algorithm.safetypotential import SafetyPotential
from algorithm.actor import Actor
from algorithm.routepredictor_DriveStyle_AvgLatent import RoutePredictor_DriveStyle_AvgLatent
from algorithm.routepredictor_DriveStyle import RoutePredictor_DriveStyle
from algorithm.routepredictor_Default import RoutePredictor_Default
from datetime import datetime

tf.disable_eager_execution()

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

log_dir = "policy_test_log/result_240313/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

agent_num = 100
actor_len = 1
try:
    world = client.get_world()


    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = True
    settings.actor_active_distance = 100000

    settings.substepping = False
    #settings.max_substeps = 5
    #settings.max_substep_delta_time = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    traffic_manager.set_synchronous_mode(True)

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
    actor = Actor(world, client)
    latcontroller = PIDLateralController(K_P =  1.5, K_I = 0.05, K_D = 0.2, dt=0.05)
    loncontroller = PIDLongitudinalController(K_P = 1.0, K_I = 0.05, K_D = 0., dt=0.05)
    sff = SafetyPotential(npc_count=agent_num)
    sess = tf.Session()
    predictor_default = RoutePredictor_Default()
    predictor_latent = RoutePredictor_DriveStyle(laneinfo, npc_count=agent_num, sess=sess, name="_latent", 
                                                 snapshot="train_log/DrivingStyle_Latent4_nextstate2/log_2023-12-28-14-55-59_1640.ckpt")
    predictor_avglatent = RoutePredictor_DriveStyle_AvgLatent(laneinfo, npc_count=agent_num, sess=sess, name="_avglatent", 
                                                 snapshot="train_log/DrivingStyle_AvgLatent4/2024-03-13-14-05-16_140.ckpt")


    predictor_desc = [
        (0, "Default", predictor_default, None),
        (2, "Latent_True", predictor_latent, True),
        (4, "AvgLatent_True", predictor_avglatent, True)
    ]
    log_file = [open(log_dir + "exp_result_" + name[1] + ".txt", "at") for name in predictor_desc]

    for a in world.get_actors():
        if isinstance(a, carla.TrafficLight):
            a.set_green_time(15)
            a.set_yellow_time(3)
            a.set_red_time(2)


    for exp in range(5000):
        distance_to_leading_vehicle = [ np.random.uniform(3.0, 10.0) for i in range(agent_num) ]
        vehicle_lane_offset = [ np.random.uniform(-0.5, 0.5) for i in range(agent_num) ]
        vehicle_speed = [ np.random.uniform(-50.0, 50.0) for i in range(agent_num) ]
        ignore_light = [ np.random.uniform(0.0, 1.0) for i in range(agent_num) ]
        desired_velocity = [ 11.1111 * (1.0 - vehicle_speed[i] / 100.0)  for i in range(agent_num) ]
        steering_ratio = [ np.clip(np.random.normal(1., 0.15), 0.7, 1.3) for i in range(agent_num) ]
        impatient_lane_change = [ np.random.uniform(10.0, 50.0) for i in range(agent_num) ]

        np.random.shuffle(spawn_points)
        
        for predictorindex, predictorname, predictor, uselatent in predictor_desc:

            state_vectors = []
            control_vectors = []
            latent_vectors = []

            np.random.seed(exp)
            random.seed(exp)
            traffic_manager.set_random_device_seed(exp)

            impatiece = [ 0.0 for i in range(agent_num) ]

            sff.Change_Log_File(log_dir + "exp_" + str(exp) + "_" + predictorname + "_sff.txt")

            print("exp " + str(exp) )
            npc_vehicles_list = []
            batch = []
            actor.reset()

            for i, transform in enumerate(spawn_points):
                if len(npc_vehicles_list) >= agent_num:
                    break
                blueprint = blueprints[i % len(blueprints)]
                if blueprint.has_attribute('color'):
                    color = blueprint.get_attribute('color').recommended_values[0]

                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = blueprint.get_attribute('driver_id').recommended_values[0]
                    blueprint.set_attribute('driver_id', driver_id)
                blueprint.set_attribute('role_name', 'autopilot')

                a = world.try_spawn_actor(blueprint, transform)
                if a != None:
                    npc_vehicles_list.append(a)


            for a in npc_vehicles_list:
                a.set_autopilot(True, 8000)

            
            sff.Assign_Player(actor.player)
            predictor.Assign_NPCS(npc_vehicles_list)
            if uselatent == None:
                predictor.Reset()
            else:
                predictor.Reset(uselatent)


            actorids_for_check = []
            for a in npc_vehicles_list:
                actorids_for_check.append(a.id)
            actorids_for_check.append(actor.player.id)
            check_failed = False


            for i, a in enumerate(npc_vehicles_list):
                traffic_manager.distance_to_leading_vehicle(a, distance_to_leading_vehicle[i] )
                traffic_manager.vehicle_lane_offset(a, vehicle_lane_offset[i])
                traffic_manager.vehicle_percentage_speed_difference(a, vehicle_speed[i])
                traffic_manager.ignore_lights_percentage(a, ignore_light[i])

            world.tick()
            world.tick()
            world.tick()
            world.tick()
            world.tick()
            accel, brake, steer = 1.0, 0.0, 0.0
            success = "TIMEOUT"
            for step in range(5000):
                if step > 5:
                    if ret["success_dest"]:
                        success = "SUCCESS"
                        break
                    ret = actor.step([accel, brake, steer])
                else:
                    ret = actor.step([0., 1., 0.])
                if ret["collision"]:
                    success = "FAIL"
                    break

                world.tick()


                state_vector = []
                control_vector = []
                latent_vector = []

                vehiclecontrols =[]
                transforms = []
                velocities = []
                trafficlights = []
                for i, a in enumerate(npc_vehicles_list):
                    tr = a.get_transform()
                    v = a.get_velocity()
                    transforms.append(tr)
                    velocities.append(v)
                    vel = np.sqrt(v.x * v.x + v.y * v.y)
                    if vel > 0.1:
                        impatiece[i] += (desired_velocity[i] - vel - 3.0) * 0.02
                    else:
                        impatiece[i] = 0.
                    if impatiece[i] < 0.:
                        impatiece[i] = 0.
                    if impatient_lane_change[i] < impatiece[i]:
                        traffic_manager.random_left_lanechange_percentage(a, (impatiece[i] - impatient_lane_change[i]) * impatiece[i] / 100.)
                        traffic_manager.random_right_lanechange_percentage(a, (impatiece[i] - impatient_lane_change[i]) * impatiece[i] / 100.)

                    px, py = 0, 0
                    tlight_state = carla.TrafficLightState.Unknown
                    try:
                        tlight = a.get_traffic_light()
                        tlight_state = tlight.get_state()
                        tlight_wps = tlight.get_stop_waypoints()
                        px, py = tlight_wps[0].transform.location.x, tlight_wps[0].transform.location.y
                    except:
                        pass
                    tstate = 1. if (tlight_state == carla.TrafficLightState.Red or 
                                    tlight_state == carla.TrafficLightState.Yellow) \
                                else 0.
                    trafficlights.append([tstate, px, py])

                    vc = a.get_control()
                    vc.steer = np.clip(vc.steer * steering_ratio[i], -1.0, 1.0)
                    vehiclecontrols.append(carla.command.ApplyVehicleControl(a, vc))

                    state_vector.append([tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y, tlight_state, [px, py]])
                    control_vector.append([impatiece[i], vc.steer, vc.throttle, vc.brake])
                    if predictorname != "Default":
                        latent_vector.append(predictor.global_latent_mean[i].tolist())

                tr = actor.player.get_transform()
                v = actor.player.get_velocity()
                transforms.append(tr)
                velocities.append(v)

                px, py = 0, 0
                tlight_state = carla.TrafficLightState.Unknown
                try:
                    tlight = actor.player.get_traffic_light()
                    tlight_state = tlight.get_state()
                    tlight_wps = tlight.get_stop_waypoints()
                    px, py = tlight_wps[0].transform.location.x, tlight_wps[0].transform.location.y
                except:
                    pass
                tstate = 1. if (tlight_state == carla.TrafficLightState.Red or 
                                tlight_state == carla.TrafficLightState.Yellow) else 0.
                trafficlights.append([tstate, 100., 0.])
                
                if step % 4 == 0:
                    predictor.Get_Predict_Result(transforms, velocities, trafficlights,impatiece )
                                            
                    tl = actor.player.get_traffic_light_state()
                    if tl == carla.TrafficLightState.Red or tl == carla.TrafficLightState.Yellow :
                        target_velocity = sff.get_target_speed(0.0, steer, predictor.pred_route, predictor.pred_prob, transforms, velocities)
                    else:
                        target_velocity = sff.get_target_speed(40.0, steer, predictor.pred_route, predictor.pred_prob, transforms, velocities)

                acceleration = loncontroller.run_step(target_velocity, ret["velocity"]) 
                if acceleration >= 0.0:
                    accel = min(acceleration, 0.75)
                    brake = 0.0
                else:
                    accel = 0.0
                    brake = min(abs(acceleration), 0.75)


                steer = latcontroller.run_step(actor.route[2][0].transform, actor.player.get_transform())

                state_vector.append([tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y, tlight_state, [px, py]])
                control_vector.append([0., steer, accel, brake])

                client.apply_batch(vehiclecontrols)

                
                state_vectors.append(state_vector)
                control_vectors.append(control_vector)
                if predictorname != "Default":
                    latent_vectors.append(latent_vector)

            
            print(str(exp) + "\t" + str(step + 1) + "\t" + success + "\n")
            log_file[predictorindex].write(str(exp) + "\t" + str(step + 1) + "\t" + success + "\n")
            client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicles_list])
            npc_vehicles_list = []

            save_obj = {}
            save_obj["params"] = [
                {"distance_to_leading_vehicle" : distance_to_leading_vehicle[i],
                "vehicle_lane_offset" : vehicle_lane_offset[i],
                "vehicle_speed" : vehicle_speed[i],
                "ignore_light" : ignore_light[i],
                "steering_ratio" : steering_ratio[i],
                "impatient_lane_change" : impatient_lane_change[i]} for i in range(agent_num)]
            save_obj["state_vectors"] = state_vectors
            save_obj["control_vectors"] = control_vectors
            if predictorname != "Default":
                save_obj["latent_vectors"] = latent_vectors
            with open(log_dir + "exp_" + str(exp) + "_" + predictorname + ".pkl","wb") as fw:
                pickle.dump(save_obj, fw)
finally:
    sff.destroy()
    actor.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in npc_vehicles_list])
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)


    time.sleep(0.5)


