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

import carla
import pygame

import numpy as np
import cv2
import random
import weakref


class HumanAgent(object):
    def __init__(self, world, client):
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
        self.player = None
        self.front_camera = None
        self.leftside_camera = None
        self.rightside_camera = None

        self.spawn_points = self.map.get_spawn_points()

        pygame.joystick.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def step(self):
        if self.player != None:

            v_vec = self.player.get_velocity()
            self.velocity = np.sqrt(v_vec.x ** 2 + v_vec.y ** 2) 

            pygame.event.get()
            numAxes = self.joystick.get_numaxes()
            jsInputs = [float(self.joystick.get_axis(i)) for i in range(numAxes)]


            self.steerCmd = 0.45 * np.tan(1.1 * jsInputs[0])

            if jsInputs[2] < -0.9:
                jsInputs[2] = -0.9
            elif jsInputs[2] > 0.9:
                jsInputs[2] = 0.9
            #self.throttleCmd = 1.6 + (2.05 * np.log10(-0.7 * jsInputs[2] + 1.4) - 1.2) / 0.92
            self.throttleCmd = ((0.9 - jsInputs[2]) / 1.8) ** 0.75

            #self.brakeCmd = 1.6 + (2.05 * np.log10(-0.7 * jsInputs[3] + 1.4) - 1.2) / 0.92
            if jsInputs[3] < -0.9:
                jsInputs[3] = -0.9
            elif jsInputs[3] > 0.9:
                jsInputs[3] = 0.9
            self.brakeCmd = ((0.9 - jsInputs[3]) / 1.8)

            control = carla.VehicleControl()
            control.throttle = self.throttleCmd * 0.7 + 0.3
            control.brake = self.brakeCmd
            control.steer = self.steerCmd
            control.manual_gear_shift = False
            control.hand_brake = False
            control.reverse = False
            control.gear = 0
            self.player.apply_control(control)

    def reset(self):
        self.destroy()
        bp = random.choice(self.blueprints)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        while self.player is None:
            spawn_point = random.choice(self.spawn_points)
            self.player = self.world.try_spawn_actor(bp, spawn_point)

        self.front_camera = Camera(self.player, 'front')
        self.leftside_camera = Camera(self.player, 'leftside')
        self.rightside_camera = Camera(self.player, 'rightside')


        physics_control = self.player.get_physics_control()
        physics_control.drag_coefficient = 2.0
        physics_control.max_rpm = 5000.0
        #physics_control.damping_rate_full_throttle = 0.25
        #physics_control.damping_rate_zero_throttle_clutch_disengaged = 5.0
        #physics_control.moi = 4.0
        self.player.apply_physics_control(physics_control)

    def render(self, image, fps):
        if self.front_camera != None:
            if self.front_camera.image is not None:
                image[:] = self.front_camera.image
        if self.leftside_camera != None:
            if self.leftside_camera.image is not None:
                im = cv2.copyMakeBorder(self.leftside_camera.image, 10, 10, 0, 20, cv2.BORDER_CONSTANT, None, value = 0)
                image[0:560, 0:560] = im
        if self.rightside_camera != None:
            if self.rightside_camera.image is not None:
                im = cv2.copyMakeBorder(self.rightside_camera.image, 10, 10, 20, 0, cv2.BORDER_CONSTANT, None, value = 0)
                image[0:560, 2000:2560] = im

        cv2.rectangle(image, (600, 1240), (1960, 1440), (0, 0, 0), -1)
        cv2.putText(image, "FPS: " + str(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        cv2.putText(image, "%3d" % int(self.velocity * 3.6), (640, 1360), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0))
 
        cv2.putText(image, "Steer : %.3f" % self.steerCmd, (1460, 1280), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255))
        cv2.putText(image, "Throttle : %.3f" % self.throttleCmd, (1460, 1330), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255))
        cv2.putText(image, "Brake : %.3f" % self.brakeCmd, (1460, 1380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255))

    def destroy(self):
        if self.front_camera != None:
            self.front_camera.destroy()
            self.front_camera = None
        if self.leftside_camera != None:
            self.leftside_camera.destroy()
            self.leftside_camera = None
        if self.rightside_camera != None:
            self.rightside_camera.destroy()
            self.rightside_camera = None
        if self.player != None:
            self.player.destroy()
            self.player = None

class Camera(object):
    def __init__(self, parent_actor, type):
        self.actor = parent_actor
        self.sensor = None
        self.image = None

        bound_x = 0.5 + self.actor.bounding_box.extent.x
        bound_y = 0.5 + self.actor.bounding_box.extent.y
        bound_z = 0.5 + self.actor.bounding_box.extent.z
        if type == 'front':
            camera_transform = carla.Transform(
                carla.Location(x=0.45*bound_x, y=-0.2*bound_y, z=1.0*bound_z),
                carla.Rotation(pitch=0.0))
            camera_dim = [2560, 1440]
            camera_fov = 110
        elif type == 'leftside':
            camera_transform = carla.Transform(
                carla.Location(x=0*bound_x, y=-0.8*bound_y, z=0.8*bound_z),
                carla.Rotation(yaw=200.0, pitch=0.0))
            camera_dim = [540, 540]
            camera_fov = 100
        elif type == 'rightside':
            camera_transform = carla.Transform(
                carla.Location(x=0*bound_x, y=0.8*bound_y, z=0.8*bound_z),
                carla.Rotation(yaw=160.0, pitch=0.0))
            camera_dim = [540, 540]
            camera_fov = 100
            
        world = parent_actor.get_world()
        bp_library = world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(camera_dim[0]))
        bp.set_attribute('image_size_y', str(camera_dim[1]))
        if bp.has_attribute('gamma'):
            bp.set_attribute('gamma', str(2.2))
        if bp.has_attribute('fov'):
            bp.set_attribute('fov', str(camera_fov))

        self.sensor = world.spawn_actor(bp, camera_transform, attach_to=parent_actor, attachment_type=carla.AttachmentType.Rigid)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: Camera._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.image = array

    def destroy(self):
        if self.sensor != None:
            self.sensor.stop()
            self.sensor.destroy()