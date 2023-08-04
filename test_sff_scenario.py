import numpy as np
import cv2
import random
import tensorflow.compat.v1 as tf
import sys
import os
import glob
import time
import math
import itertools

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

def Get_spawn_point(spawn_points, x, y):
    for s in spawn_points:
        if -1 < s.location.x - x < 1 and -1 < s.location.y - y < 1:
            return s

agent_num = 1
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

    npc_spawn_point = Get_spawn_point(spawn_points, -1.55, -180.80)
    agent_spawn_point = Get_spawn_point(spawn_points, -5.25, -214.15)
    agent_destination_point = Get_spawn_point(spawn_points, -5.15, 128.50)


    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    
    actor = Actor(world, client, blueprints=[blueprints[0]])
    latcontroller = PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05)
    loncontroller = PIDLongitudinalController(K_P = 1.0, K_I = 0.05, K_D = 0., dt=0.05)
    sff = SafetyPotential(laneinfo=laneinfo, visualize=True, record_video=False, agent_count=agent_num)

    tf.disable_eager_execution()
    sess = tf.Session()
    with sess.as_default():
        for npc_velocity, npc_start_distance, npc_lane_offset in itertools.product([20, 24, 28], [7.5, 9., 10.5, 12., 13.5], [-0.5, -0.25, 0., 0.25, 0.5]):
            for distance in [("Default", 0), ("DriveStyle", 0), ("DriveStyle", 64), ("DriveStyle", 256)]:
                sff.set_global_distance(distance[0], distance[1])

                distance_to_leading_vehicle = [ np.random.uniform(3.0, 10.0) for i in range(agent_num) ]
                vehicle_lane_offset = [ np.random.uniform(-0.5, 0.5) for i in range(agent_num) ]
                vehicle_speed = [ np.random.uniform(-50.0, 50.0) for i in range(agent_num) ]
                ignore_light = [ np.random.uniform(0.0, 1.0) for i in range(agent_num) ]
                desired_velocity = [ 11.1111 * (1.0 - vehicle_speed[i] / 100.0)  for i in range(agent_num) ]

                log_file = open("policy_test_log/result_0801/zsff_policy_" + distance[0] + "_" + str(distance[1]) \
                                + "_" + str(npc_velocity) + "_" + str(npc_start_distance) + "_" + str(npc_lane_offset) + ".txt", "wt")

                print("exp " + str(npc_velocity) + "_" + str(npc_start_distance) + "_" + str(npc_lane_offset) )
                vehicles_list = []
                batch = []
                actor.spawn_point = agent_spawn_point
                actor.dest_point = agent_destination_point
                actor.reset()

                impatiece = [ 0.0 for i in range(agent_num) ]

                transform = npc_spawn_point
                blueprint = blueprints[0]
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
                    traffic_manager.vehicle_lane_offset(a, npc_lane_offset)
                    traffic_manager.auto_lane_change(a, False)
                    traffic_manager.ignore_vehicles_percentage(a, 100)
                    traffic_manager.vehicle_percentage_speed_difference(a, 100 - (npc_velocity / 30) * 100)

                world.tick()
                world.tick()
                world.tick()
                world.tick()
                world.tick()
                success = 0
                accel, brake, steer = 1.0, 0.0, 0.0
                lane_changed = False
                for step in range(500):
                    ret = actor.step([accel, brake, steer])
                    world.tick()
                    if ret["collision"]:
                        break
                    if ret["success_dest"]:
                        success += 1

                    npc_tr = all_vehicle_actors[0].get_transform()
                    agent_tr = actor.player.get_transform()
                    if lane_changed == False and npc_tr.location.y - agent_tr.location.y < npc_start_distance:
                        traffic_manager.force_lane_change(all_vehicle_actors[0], True)
                        lane_changed = True
                    '''
                    tl = actor.player.get_traffic_light_state()
                    print(tl)
                    if tl == carla.TrafficLightState.Red:
                        target_velocity, log = sff.get_target_speed(0.0, print_log=True, impatience=impatiece)
                    else:
                        target_velocity, log = sff.get_target_speed(40.0, print_log=True, impatience=impatiece)
                    '''
                    target_velocity, log = sff.get_target_speed(40.0, print_log=True, impatience=impatiece)
                    acceleration = loncontroller.run_step(target_velocity, ret["velocity"]) 
                    if acceleration >= 0.0:
                        accel = min(acceleration, 0.75)
                        brake = 0.0
                    else:
                        accel = 0.0
                        brake = min(abs(acceleration), 0.75)


                    steer = latcontroller.run_step(actor.route[2][0].transform, actor.player.get_transform())
                    log_file.write(str(step + 1) + "\t" + str(target_velocity) + "\t" + log + "\n")
                    print(log)

                print( str(step + 1) + "\t" + str(success) + "\n")
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


