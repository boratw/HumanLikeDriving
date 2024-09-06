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


zero_image = np.full((640, 1280), 0., dtype=np.float32)
one_image = np.full((640, 1280), 1., dtype=np.float32)

class Actor(object):
    def __init__(self, world, client, action_ratio = 20.0, route_plan_hop_resolution=2.0, forward_target=5.0, target_velocity=4.0, 
                 min_waypoint_distance=3.0, blueprints=None, spawn_point=None, dest_point=None):
        self.world = world
        self.client = client
        self.forward_target = forward_target
        self.target_velocity = target_velocity
        self.min_waypoint_distance = min_waypoint_distance
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.collision_sensor = None
        self.camera = None
        self.action_ratio = action_ratio
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
        self.npc_vehicle_list = []


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


    def reset(self):
        self.destroy()

        bp = random.choice(self.blueprints)
        if self.spawn_point == None:
            spawn_points = self.map.get_spawn_points()
            spawn_point_indices = list(range(len(spawn_points)))
            #spawn_point_indices = sorted(spawn_point_indices, key=lambda i : (spawn_points[i].location.x ** 2 + spawn_points[i].location.y ** 2))
            while self.player is None:
                spawn_index = random.choice(spawn_point_indices)
                spawn_point = spawn_points[spawn_index]
                self.player = self.world.try_spawn_actor(bp, spawn_point)
            spawn_point_indices.remove(spawn_index)
        else:
            self.player = self.world.try_spawn_actor(bp, self.spawn_point)


        self.collision_sensor = CollisionSensor(self.player)
        #self.camera = Camera(self.player)

        self.route.clear()
        
        

    def new_destination(self):
        self.route.clear()
        vehicle_transform = self.player.get_transform()
        ego_loc = vehicle_transform.location
        f_vec = vehicle_transform.get_forward_vector()
        spawn_points = self.map.get_spawn_points()
        if self.dest_point == None:
            destination = random.choice(spawn_points).location
        else:
            destination = self.dest_point.location

        #start_waypoint = self.map.get_waypoint(ego_loc)
        #end_waypoint = self.map.get_waypoint(destination)
        #route = self.route_planner.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        route = self.route_planner.trace_route(ego_loc, destination)
        for r in route:
            self.route.append(r)

        w_norm = [f_vec.x, f_vec.y]
        self.prev_w_norm = w_norm
    
    def step(self, action, return_control=False):


        vehicle_transform = self.player.get_transform()
        ego_loc = vehicle_transform.location
        v_vec = self.player.get_velocity()
        f_vec = vehicle_transform.get_forward_vector()
        predict_loc = ego_loc + f_vec * 3.0

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

        w_vec = self.route[2][0].transform.location - ego_loc

        theta = np.arctan2(w_vec.y, w_vec.x) - np.arctan2(f_vec.y, f_vec.x)
        if theta > np.pi:
            theta -= (2 * np.pi)
        elif theta < -np.pi:
            theta += (2 * np.pi)

        control = carla.VehicleControl()
        control.throttle = action[0]
        control.brake = action[1]
        control.steer = action[2]
        control.manual_gear_shift = False
        control.hand_brake = False
        control.reverse = False
        control.gear = 0
        if return_control :
            return {
                "velocity" : np.sqrt(v_vec.x ** 2 + v_vec.y ** 2),
                "dest_angle" : theta,
                "success_dest" : success_dest,
                "collision" : self.collision_sensor.collision,
                "control" : control
            }
        else :
            self.player.apply_control(control)
            return {
                "velocity" : np.sqrt(v_vec.x ** 2 + v_vec.y ** 2),
                "dest_angle" : theta,
                "success_dest" : success_dest,
                "collision" : self.collision_sensor.collision
            }

    

    def destroy(self):
        if self.collision_sensor != None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.camera != None:
            self.camera.destroy()
            self.camera = None
        if self.player != None:
            self.player.destroy()
            self.player = None


    @staticmethod
    def get_state_len():
        return 8

    @staticmethod
    def get_action_len():
        return 4

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
    
    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = np.array([sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z])
        self.gyroscope = np.array([sensor_data.gyroscope.x, sensor_data.gyroscope.y, sensor_data.gyroscope.z])
        '''
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        '''
        self.compass = math.degrees(sensor_data.compass)


    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.collision = False
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision = True

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()

class Camera(object):
    def __init__(self, parent_actor, gamma_correction=2.2, width=1280, height=720):
        self.sensor = None
        self._parent = parent_actor
        self.image = np.zeros((height, width, 3), dtype=np.dtype("uint8"))
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        if bp.has_attribute('gamma'):
            bp.set_attribute('gamma', str(gamma_correction))

        #transform = carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=8.0))
        attachment = carla.AttachmentType.Rigid
        self.sensor = world.spawn_actor(bp, transform, attach_to=self._parent, attachment_type=attachment)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Camera._parse_image(weak_self, image))


    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.image = array

    
    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


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
