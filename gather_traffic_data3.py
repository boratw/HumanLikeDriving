#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

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

import numpy as np

from carla import VehicleLightState as vls
from carla import TrafficLightState as tls

import random
import pickle

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


vehicles_list = []
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

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

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------

    for exp in range(1000):
        save_objs = []

        distance_to_leading_vehicle = [ np.random.uniform(3.0, 10.0) for i in range(agent_num) ]
        vehicle_lane_offset = [ np.random.uniform(-0.5, 0.5) for i in range(agent_num) ]
        vehicle_speed = [ np.random.uniform(-50.0, 50.0) for i in range(agent_num) ]
        ignore_light = [ np.clip(np.random.normal(0., 2.), 0., 10.0) for i in range(agent_num) ]
        ignore_vehicle = [ np.clip(np.random.normal(0., 1.), 0., 5.0) for i in range(agent_num) ]
        steering_ratio = [ np.clip(np.random.normal(1., 0.15), 0.7, 1.3) for i in range(agent_num) ]

        desired_velocity = [ 11.1111 * (1.0 - vehicle_speed[i] / 100.0)  for i in range(agent_num) ]

        impatient_lane_change = [ np.random.uniform(10.0, 160.0) for i in range(agent_num) ]
        
        for iteration in range(50):
            print("exp " + str(exp) + " : " + str(iteration))
            random.shuffle(spawn_points)
            vehicles_list = []
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
                    vehicles_list.append(response.actor_id)

            all_vehicle_actors = world.get_actors(vehicles_list)

            for i, actor in enumerate(all_vehicle_actors):
                traffic_manager.distance_to_leading_vehicle(actor, distance_to_leading_vehicle[i] )
                traffic_manager.vehicle_lane_offset(actor, vehicle_lane_offset[i])
                traffic_manager.vehicle_percentage_speed_difference(actor, vehicle_speed[i])
                traffic_manager.ignore_lights_percentage(actor, ignore_light[i])
                traffic_manager.ignore_lights_percentage(actor, ignore_vehicle[i])


                
            steer_add = [ 0. for _ in range(agent_num) ]
            throttle_add = [ 0. for _ in range(agent_num) ]
            brake_add = [ 0. for _ in range(agent_num) ]
            torque_added = [ 0 for _ in range(agent_num) ]
            
            impatiece = [ 0.0 for i in range(agent_num) ]

            world.tick()
            for step in range(2000):
                state_vector = []
                control_vector = []
                vehiclecontrols = []
                for i, actor in enumerate(all_vehicle_actors):


                    tr = actor.get_transform()
                    v = actor.get_velocity()
                    try:
                        tlight = actor.get_traffic_light()
                        tlight_state = tlight.get_state()
                        tlight_wps = tlight.get_stop_waypoints()
                        tlight_pos = [[w.transform.location.x, w.transform.location.y] for w in tlight_wps ]
                    except:
                        tlight_state = carla.TrafficLightState.Unknown
                        tlight_pos = []

                    fail = actor.get_failure_state()
                    try:
                        traj = traffic_manager.get_all_actions(actor)
                        traj_pos = [ [t[0], t[1].transform.location.x, t[1].transform.location.y, t[1].transform.rotation.yaw] for t in traj ]
                    except:
                        traj_pos = []
                    
                    vel = np.sqrt(v.x * v.x + v.y * v.y)
                    if vel > 0.1:
                        impatiece[i] += (desired_velocity[i] - vel - 3.0) * 0.02
                    else:
                       impatiece[i] = 0.
                    if impatiece[i] < 0.:
                        impatiece[i] = 0.
                    
                    if impatient_lane_change[i] < impatiece[i]:
                        traffic_manager.random_left_lanechange_percentage(actor, (impatiece[i] - impatient_lane_change[i]) * 0.1)
                        traffic_manager.random_right_lanechange_percentage(actor, (impatiece[i] - impatient_lane_change[i]) * 0.1)



                    if random.random() < (1 / 200):
                        torque_added[i] = 5
                        steer_add[i] = random.random() - 0.5
                        throttle_add[i] = random.random() * 0.5
                        brake_add[i] = random.random() * 0.5

                    vc = actor.get_control()
                    if torque_added[i] >= 1:
                        vc.steer = np.clip(vc.steer + steer_add[i], -1.0, 1.0)
                        vc.throttle = np.clip(vc.throttle + throttle_add[i], 0.0, 1.0)
                        vc.brake = np.clip(vc.brake + brake_add[i], 0.0, 1.0)
                        torque_added[i] -= 1
                    else:
                        vc.steer = np.clip(vc.steer * steering_ratio[i], -1.0, 1.0)


                    vehiclecontrols.append(carla.command.ApplyVehicleControl(actor, vc))

                    state = [tr.location.x, tr.location.y, tr.rotation.yaw, v.x, v.y, tlight_state, tlight_pos, fail, traj_pos]
                    control = [torque_added[i], impatiece[i], vc.steer, vc.throttle, vc.brake]

                    state_vector.append(state)
                    control_vector.append(control)

                client.apply_batch(vehiclecontrols)
                world.tick()
                state_vectors.append(state_vector)
                control_vectors.append(control_vector)
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])


            save_obj = {}
            save_obj["params"] = [ [distance_to_leading_vehicle[i], vehicle_lane_offset[i], vehicle_speed[i], impatient_lane_change[i], steering_ratio[i], ignore_light[i], ignore_vehicle[i]] for i in range(agent_num) ]
            save_obj["state_vectors"] = state_vectors
            save_obj["control_vectors"] = control_vectors
            save_objs.append(save_obj)
        with open("data/gathered_from_npc_batjeon6/data_" + str(exp) + ".pkl","wb") as fw:
            pickle.dump(save_objs, fw)


finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    print('\ndestroying %d vehicles' % len(vehicles_list))
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

    time.sleep(0.5)