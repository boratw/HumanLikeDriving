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
from laneinfo import LaneInfo
from algorithm.controller import PIDLongitudinalController, PIDLateralController
from algorithm.safetypotential import SafetyPotential
from algorithm.actor import Actor

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

agent_num = 150
try:
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
    settings.no_rendering_mode = True
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

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    
    actor = Actor(world, client, blueprints=blueprints)
    latcontroller = PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05)
    loncontroller = PIDLongitudinalController(K_P = 1.0, K_I = 0.05, K_D = 0., dt=0.05)
    sff = SafetyPotential(laneinfo=laneinfo, visualize=False, record_video=False, agent_count=agent_num)

    tf.disable_eager_execution()
    sess = tf.Session()
    with sess.as_default():
        for exp in range(1000):
            distance_to_leading_vehicle = [ np.random.uniform(3.0, 10.0) for i in range(agent_num) ]
            vehicle_lane_offset = [ np.random.uniform(-0.5, 0.5) for i in range(agent_num) ]
            vehicle_speed = [ np.random.uniform(-50.0, 50.0) for i in range(agent_num) ]
            ignore_light = [ np.random.uniform(0.0, 1.0) for i in range(agent_num) ]
            desired_velocity = [ 11.1111 * (1.0 - vehicle_speed[i] / 100.0)  for i in range(agent_num) ]

            impatient_lane_change = [ np.random.uniform(10.0, 50.0) for i in range(agent_num) ]
            impatiece = [ 0.0 for i in range(agent_num) ]

            log_file = open("policy_test_log/sff_policy_drivestyle_" + str(exp) + ".txt", "wt")
            log_file.write("Iteration\tSurvive_Time\tScore\n")

            print("exp " + str(exp) )
            random.shuffle(spawn_points)
            vehicles_list = []
            batch = []
            actor.reset()

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
                    vehicles_list.append(response.actor_id)

            all_vehicle_actors = world.get_actors(vehicles_list)
            
            sff.Assign_Player(actor.player)
            sff.Assign_NPCS(all_vehicle_actors)

            for i, a in enumerate(all_vehicle_actors):
                traffic_manager.distance_to_leading_vehicle(a, distance_to_leading_vehicle[i] )
                traffic_manager.vehicle_lane_offset(a, vehicle_lane_offset[i])
                traffic_manager.vehicle_percentage_speed_difference(a, vehicle_speed[i])
                traffic_manager.ignore_lights_percentage(a, ignore_light[i])

            world.tick()
            world.tick()
            world.tick()
            world.tick()
            world.tick()
            success = 0
            accel, brake, steer = 1.0, 0.0, 0.0
            for step in range(2000):
                ret = actor.step([accel, brake, steer])
                world.tick()
                if ret["collision"]:
                    break
                if ret["success_dest"]:
                    success += 1

                for i, a in enumerate(all_vehicle_actors):
                    v = a.get_velocity()
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

                target_velocity, log = sff.get_target_speed(40.0, print_log=True, impatience=impatiece)

                acceleration = loncontroller.run_step(target_velocity, ret["velocity"]) 
                if acceleration >= 0.0:
                    accel = min(acceleration, 0.75)
                    brake = 0.0
                else:
                    accel = 0.0
                    brake = min(abs(acceleration), 0.3)


                steer = latcontroller.run_step(actor.route[2][0].transform, actor.player.get_transform())
                log_file.write(str(step + 1) + "\t" + str(target_velocity) + "\t" + log + "\n")
                print(log)

            print(str(exp) + "\t" + str(step + 1) + "\t" + str(success) + "\n")
            #log_file.write(str(exp) + "\t" + str(step + 1) + "\t" + str(success) + "\n")
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
            vehicles_list = []
finally:
    sff.destroy()
    actor.destroy()
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)


    time.sleep(0.5)


