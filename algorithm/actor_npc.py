import glob
import os
import sys
import random
import time
import numpy as np
import math
import weakref
from collections import deque

try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import cv2
from carla import ColorConverter as cc
from algorithm.global_route_planner import GlobalRoutePlanner, RoadOption
from algorithm.controller import PIDLongitudinalController, PIDLateralController
from algorithm.vehicle_detector import detect_vehicles



zero_image = np.full((640, 1280), 0., dtype=np.float32)
one_image = np.full((640, 1280), 1., dtype=np.float32)

class Actor_NPC(object):
    def __init__(self, world, client, route_plan_hop_resolution=2.0, 
                 min_waypoint_distance=3.0, blueprints=None, spawn_point=None, dest_point=None,
                 target_velocity=15.0, ):
        self.world = world
        self.client = client
        self.min_waypoint_distance = min_waypoint_distance

        self.target_velocity = target_velocity
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.route_planner = GlobalRoutePlanner(self.map, route_plan_hop_resolution)
        self.route_planner.setup()
        self.route = deque(maxlen=1000)
        self.player = None



        if blueprints == None:
            blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]
        self.blueprints = blueprints
        self.spawn_point = spawn_point
        self.dest_point = dest_point

        self.latcontroller = PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05)
        self.loncontroller = PIDLongitudinalController(K_P = 1.0, K_I = 0.05, K_D = 0., dt=0.05)

        if self.spawn_point == None:
            self.spawn_point = self.map.get_spawn_points()

    def reset(self):
        self.destroy()

        bp = random.choice(self.blueprints)
        while self.player is None:
            spawn_point = random.choice(self.spawn_point)
            self.player = self.world.try_spawn_actor(bp, spawn_point)


        self.route.clear()
        
    def assign_others(self, others):
        self.others = others

    def new_destination(self):
        self.route.clear()
        vehicle_transform = self.player.get_transform()
        ego_loc = vehicle_transform.location
        f_vec = vehicle_transform.get_forward_vector()
        if self.dest_point == None:
            destination = random.choice(self.spawn_point).location
        else:
            destination = self.dest_point.location

        start_waypoint = self.map.get_waypoint(ego_loc)
        end_waypoint = self.map.get_waypoint(destination)
        route = self.route_planner.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        for r in route:
            self.route.append(r)

        w_norm = [f_vec.x, f_vec.y]
        self.prev_w_norm = w_norm
    
    def step(self):


        vehicle_transform = self.player.get_transform()
        ego_loc = vehicle_transform.location
        v_vec = self.player.get_velocity()
        f_vec = vehicle_transform.get_forward_vector()
        predict_loc = ego_loc + f_vec * 3.0

        self.actor_tr = vehicle_transform
        self.actor_v = v_vec

        max_index = -1
        prev_d = None
        for i, (waypoint, roadoption) in enumerate(self.route):
            d = waypoint.transform.location.distance(predict_loc)
            if prev_d != None:
                if prev_d > d:
                    max_index = i
            if i == 10:
                break
            prev_d = d
                
        if max_index >= 0:
            for i in range(max_index + 1):
                self.route.popleft()

        success_dest = (len(self.route) < 5)
        while len(self.route) < 5:
            self.new_destination()
            if len(self.route) > 5:
                max_index = -1
                prev_d = None
                for i, (waypoint, roadoption) in enumerate(self.route):
                    d = waypoint.transform.location.distance(predict_loc)
                    if prev_d != None:
                        if prev_d > d:
                            max_index = i
                    if i == 10:
                        break
                    prev_d = d
                        
                if max_index >= 0:
                    for i in range(max_index + 1):
                        self.route.popleft()

        velocity = np.sqrt(v_vec.x ** 2 + v_vec.y ** 2)
        detect_result = detect_vehicles(self.player, self.map, self.route, self.others, 5.0 + 3.0 * velocity)

        target_velocity = self.target_velocity

        self.tlight_state = carla.TrafficLightState.Unknown
        tlight = self.player.get_traffic_light()
        if tlight:
            self.tlight_state = tlight.get_state()
            if self.tlight_state == carla.TrafficLightState.Red:
                target_velocity = 0.

        if(detect_result[0]):
            des_velocity = (detect_result[2] - 5) * 0.333333 - 2.
            if des_velocity < 0.:
                des_velocity = 0.
            if target_velocity > des_velocity:
                target_velocity = des_velocity
        

        acceleration = self.loncontroller.run_step(target_velocity, velocity) 
        if acceleration >= 0.0:
            accel = min(acceleration, 0.75)
            brake = 0.0
        else:
            accel = 0.0
            brake = min(abs(acceleration), 0.75)
        
        steer = self.latcontroller.run_step(self.route[4][0].transform, vehicle_transform)


        control = carla.VehicleControl()
        control.throttle = accel
        control.brake = brake
        control.steer = steer
        control.manual_gear_shift = False
        control.hand_brake = False
        control.reverse = False
        control.gear = 0
        return control


    def destroy(self):
        if self.player != None:
            self.player.destroy()
            self.player = None


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
