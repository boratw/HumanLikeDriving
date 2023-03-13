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

import re
from classic_agent import ClassicAgent
from map_reader import LaneInfo

  
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

    client = carla.Client("localhost", 2000) 
    client.set_timeout(2.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 20
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    world.set_weather(find_weather('Clear Noon'))


    laneinfo = LaneInfo(world)
    npcs = [ClassicAgent(world, client, laneinfo) for _ in range(1)]

    for iter in range(1000):
        for npc in npcs:
            npc.reset()

        for npc in npcs:
            npc.assign_others([n.player for n in npcs if n != npc])

        for step in range(2000):
            for npc in npcs:
                npc.step()
            world.tick()

finally:
    
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)
    for npc in npcs:
        npc.destroy()
    print("All cleaned up!")


