
import carla

import numpy as np
import cv2
import random
import weakref

from map_reader import RoutePlanner
from algorithm.controller import PIDLongitudinalController, PIDLateralController
from algorithm.vehicle_detector import detect_vehicles

class ClassicAgent(object):
    def __init__(self, world, client, laneinfo):
        self.world = world
        self.client = client
        self.map = self.world.get_map()

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
        self.blueprints = blueprints
        self.spawn_points = self.map.get_spawn_points()
        self.player = None
        self.route = RoutePlanner(laneinfo)

        self.latcontroller = PIDLateralController(K_P =  1.95, K_I = 0.05, K_D = 0.2, dt=0.05)
        self.loncontroller = PIDLongitudinalController(K_P = 1.0, K_I = 0.05, K_D = 0., dt=0.05)

    def reset(self):
        self.destroy()
        self.route.Clear()
        bp = random.choice(self.blueprints)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        while self.player is None:
            spawn_point = random.choice(self.spawn_points)
            self.player = self.world.try_spawn_actor(bp, spawn_point)

    def step(self):
        vehicle_transform = self.player.get_transform()
        ego_loc = vehicle_transform.location
        self.route.Move(ego_loc)
        v_vec = self.player.get_velocity()
        velocity = np.sqrt(v_vec.x ** 2 + v_vec.y ** 2)

        if len(self.route.route) > 5:
            detect_result = detect_vehicles(self.player, self.map, self.route.route, self.others, 8.0 + 0.5 * velocity)

            target_velocity = 15.0
            if(detect_result[0]):
                target_velocity = detect_result[2] * 0.5 - 1.5
                if target_velocity < 0.:
                    target_velocity = 0.
                elif target_velocity > 15.:
                    target_velocity = 15. 

            acceleration = self.loncontroller.run_step(target_velocity, velocity) 
            if acceleration >= 0.0:
                accel = min(acceleration, 0.75)
                brake = 0.0
            else:
                accel = 0.0
                brake = min(abs(acceleration), 0.75)
            steer = self.latcontroller.run_step(self.route.route[5], vehicle_transform)

            control = carla.VehicleControl()
            control.throttle = accel
            control.brake = brake
            control.steer = steer
            control.manual_gear_shift = False
            control.hand_brake = False
            control.reverse = False
            control.gear = 0

            self.player.apply_control(control)

    def destroy(self):
        if self.player != None:
            self.player.destroy()
            self.player = None

    def assign_others(self, others):
        self.others = others