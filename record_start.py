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

from carla import ColorConverter as cc
import numpy as np
import cv2
import random
import pygame

import re
from human_agent import HumanAgent

  
def find_weather(target = None):
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    list = [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

    item = None
    for i in list:
        if i[1] == target:
            item = i[0]
            break
    if item == None:
        return random.choice(list)[0]
    else:
        return item

        

try:
    pygame.init()
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000) 
    client.set_timeout(2.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 20
    settings.synchronous_mode = True
    settings.no_rendering_mode = True
    world.apply_settings(settings)
    world.set_weather(find_weather('Clear Noon'))


    hero = HumanAgent(world, client)
    hero.reset()
    display = np.zeros((1440, 2560, 3), dtype=np.uint8)
    while True:
        hero.step()
        hero.render(display, clock.get_fps())

        clock.tick_busy_loop(20)
        world.tick()
        cv2.imshow("game", display)
        if cv2.waitKey(1) == 27:
            break

finally:
    pygame.quit()
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    hero.destroy()
    print("All cleaned up!")


