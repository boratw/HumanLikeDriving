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
from laneinfo import LaneInfo



vehicles_list = []
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

try:
    world = client.get_world()
    laneinfo = LaneInfo()
    laneinfo.Load_from_World(world)
    laneinfo.Save("laneinfo_World10Opt.pkl")



finally:
    time.sleep(0.5)