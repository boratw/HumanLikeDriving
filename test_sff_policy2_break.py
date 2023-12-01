import numpy as np
import cv2
import random
import tensorflow.compat.v1 as tf
import sys
import os
import glob
import time
import math
import argparse
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
from algorithm.safetypotential_DriveStyle11 import SafetyPotential
from algorithm.actor import Actor
from algorithm.actor_skill import ActorSkill

parser = argparse.ArgumentParser()
parser.add_argument('--exp', dest='exp', action='store')
args = parser.parse_args()
input_exp = int(args.exp)

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

agent_num = 100
try:
    world = client.get_world()

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    traffic_manager.set_respawn_dormant_vehicles(False)
    traffic_manager.set_synchronous_mode(True)

    settings = world.get_settings()
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = True
    settings.actor_active_distance = 200000
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

    
    actor = Actor(world, client, blueprints=blueprints)
    latcontroller = PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05)
    loncontroller = PIDLongitudinalController(K_P = 1.0, K_I = 0.05, K_D = 0., dt=0.05)
    sff = SafetyPotential(laneinfo=laneinfo, visualize=True, record_video=True, agent_count=agent_num)
    actionskill = ActorSkill()

    tf.disable_eager_execution()
    sess = tf.Session()
    with sess.as_default():
        for exp in [input_exp]:
            try:
                distance_to_leading_vehicle = [ 15. for i in range(agent_num) ]
                vehicle_lane_offset = [ 0. for i in range(agent_num) ]
                vehicle_speed = [ np.random.uniform(-25.0, 25.0) for i in range(agent_num) ]
                ignore_light = [ 0. for i in range(agent_num) ]
                desired_velocity = [ 11.1111 * (1.0 - vehicle_speed[i] / 100.0)  for i in range(agent_num) ]
                steering_ratio = [ 0. for i in range(agent_num) ]

                impatient_lane_change = [ np.random.uniform(20.0, 100.0) for i in range(agent_num) ]
                impatiece = [ 0.0 for i in range(agent_num) ]

                log_file = open("policy_test_log/result_1128/sff_b_policy_" + str(exp) + ".txt", "wt")
                log_file_other = open("policy_test_log/result_1128/sff_b_other_" + str(exp) + ".txt", "wt")
                log_file.write("Step\tMoved Distance\tCurrent Pos_x\tCurrent Pos_y\tCurrent_Yaw\tCurrent Velocity\tTarget Velocity\tOutput_Accel\tOutput_Brake\tOutput_Steer")
                log_file.write(sff.log_caption)
                log_file.write("\n")
                log_file_other.write("Current Pos_x\tCurrent Pos_y\tCurrent_Yaw\tCurrent Velocity\n")

                print("exp " + str(exp) )
                random.shuffle(spawn_points)
                vehicles_list = []
                vehicle_params = []
                all_vehicle_actors = []
                batch = []
                actor.reset()

                for n, transform in enumerate(spawn_points):
                    if len(all_vehicle_actors) >= agent_num:
                        break
                    blueprint = random.choice(blueprints)
                    if blueprint.has_attribute('color'):
                        color = random.choice(blueprint.get_attribute('color').recommended_values)

                        blueprint.set_attribute('color', color)
                    if blueprint.has_attribute('driver_id'):
                        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                        blueprint.set_attribute('driver_id', driver_id)
                    blueprint.set_attribute('role_name', 'autopilot')

                    a = world.try_spawn_actor(blueprint, transform)
                    if a is not None:
                        all_vehicle_actors.append(a)
                        a.set_autopilot(True)


                for i, a in enumerate(all_vehicle_actors):
                    vehicles_list.append(a.id)
                    vehicle_params.append([distance_to_leading_vehicle[i], vehicle_lane_offset[i], vehicle_speed[i], impatient_lane_change[i], steering_ratio[i], ignore_light[i], 0.])
                if exp % 5 == 0:
                    sff.Assign_Player(actor.player, record_video_name = "policy_test_log/result_1128/sff_b_policy_" + str(exp) + ".avi")
                else:
                    sff.Assign_Player(actor.player, None)
                sff.Assign_NPCS(all_vehicle_actors, vehicle_params)

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
                moved = 0
                accel, brake, steer = 1.0, 0.0, 0.0
                prev_yaw = 0.
                for step in range(20000):
                    ret = actor.step([accel, brake, steer])
                    world.tick()
                    if ret["collision"]:
                        break
                    if ret["success_dest"]:
                        success += 1


                    for i in range(agent_num):
                        if all_vehicle_actors[i].is_alive == False:
                            blueprint = random.choice(blueprints)
                            for n, transform in enumerate(spawn_points):
                                a = world.try_spawn_actor(blueprint, transform)
                                print ("Try spawn actor")
                                if a is not None:
                                    all_vehicle_actors[i] = a
                                    vehicles_list[i] = a.id
                                    break


                    vehiclecontrols =[]
                    for i, a in enumerate(all_vehicle_actors):
                        tr = a.get_transform()
                        v = a.get_velocity()
                        vel = np.sqrt(v.x * v.x + v.y * v.y)

                        #vc = a.get_control()
                        #vc.steer = np.clip(vc.steer * steering_ratio[i], -1.0, 1.0)
                        #vehiclecontrols.append(carla.command.ApplyVehicleControl(a, vc))

                        log_file_other.write(str(tr.location.x) + "\t" + str(tr.location.y) + "\t" + str(tr.rotation.yaw) + "\t" + str(vel) + "\t")
                    log_file_other.write("\n")

                    tl = actor.player.get_traffic_light_state()
                    print(tl)
                    if tl == carla.TrafficLightState.Red:
                        target_velocity, log = sff.get_target_speed(0.0, steer, print_log=True, impatience=impatiece, action_latent=actionskill.z_a)
                    else:
                        target_velocity, log = sff.get_target_speed(40.0, steer, print_log=True, impatience=impatiece, action_latent=actionskill.z_a)

                    acceleration = loncontroller.run_step(target_velocity, ret["velocity"]) 
                    if acceleration >= 0.0:
                        accel = min(acceleration, 0.75)
                        brake = 0.0
                    else:
                        accel = 0.0
                        brake = min(abs(acceleration), 0.75)

                    tr = actor.player.get_transform()
                    v = actor.player.get_velocity()
                    v = np.sqrt(v.x ** 2 + v.y ** 2)
                    moved += v * 0.05
                    steer = latcontroller.run_step(actor.route[2][0].transform, tr)
                    if moved > 500:
                        if exp % 4 == 0:
                            brk = 0.25
                        elif exp % 4 == 1:
                            brk = -0.25
                        elif exp % 4 == 2:
                            brk = 0.3
                        else:
                            brk = -0.3
                    else:
                        brk = 0
                    steer += brk
                    steer = actionskill.adjust_action(steer)

                    log_file.write(str(step + 1) + "\t" + str(moved) + "\t" + str(tr.location.x) + "\t" + str(tr.location.y) + "\t" + str(tr.rotation.yaw) + "\t"
                                + str(v) + "\t" + str(target_velocity) + "\t" + str(accel) + "\t" + str(brake) + "\t" + str(steer)  + "\t" + str(actionskill.z_a) 
                                + "\t" + log + "\n")

                    yawdiff = tr.rotation.yaw - prev_yaw
                    if yawdiff > 180:
                        yawdiff -= 360
                    elif yawdiff < -180:
                        yawdiff += 360
                    if moved > 100:
                        actionskill.inference(steer, yawdiff, v, brk)
                    print("ACTION", actionskill.yaw_buffer, actionskill.steer_buffer)
                    prev_yaw = tr.rotation.yaw

                    client.apply_batch_sync(vehiclecontrols)
                    print(step, moved, v, log)
                    if moved > 2000:
                        break
            finally:
                print(str(exp) + "\t" + str(step + 1) + "\t" + str(success) + "\n")
                #log_file.write(str(exp) + "\t" + str(step + 1) + "\t" + str(success) + "\n")
                client.apply_batch_sync([carla.command.DestroyActor(x) for x in vehicles_list])
                sff.destroy()
                actor.destroy()
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


