
import carla
import numpy as np
import random
from enum import Enum
from collections import deque
import pickle

class LaneInfo(object):
    def __init__(self):
        self.lanes = {}

    def Load_from_World(self, world):
        self.map = world.get_map()
        waypoints = self.map.generate_waypoints(2.0)
        self.lanes = {}

        for w in waypoints:
            laneid = w.road_id * 100 + w.lane_id
            if laneid not in self.lanes:
                lane = {"pos" : [], "orientation" : None, "left" : None, "right" : None, "type" : LaneType.Follow, "next" : []}
                lane["pos"].append([w.transform.location.x, w.transform.location.y])
                lanelist = [w]
                for wp in w.previous_until_lane_start(2.0):
                    if wp.road_id == w.road_id:
                        lane["pos"].append([wp.transform.location.x, wp.transform.location.y])
                        lanelist.append(wp)
                lane["pos"] = lane["pos"][::-1]
                for wp in w.next_until_lane_end(2.0):
                    if wp.road_id == w.road_id:
                        lane["pos"].append([wp.transform.location.x, wp.transform.location.y])
                        lanelist.append(wp)
                lane["orientation"] = lanelist[len(lanelist) // 2].transform.rotation.yaw

                if wp.is_junction == True:
                    turning = np.sin((lanelist[-1].transform.rotation.yaw - lanelist[0].transform.rotation.yaw) * 0.017453293)
                    if turning < -0.5:
                        lane["type"] = LaneType.Left
                    elif turning > 0.5:
                        lane["type"] = LaneType.Right
                    else:
                        lane["type"] = LaneType.Straight
                else:
                    for wp in lanelist:
                        if lane["right"] == None and wp.right_lane_marking.lane_change & carla.LaneChange.Right:
                            next_waypoint = wp.get_right_lane()
                            if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and wp.road_id == next_waypoint.road_id:
                                lane["right"] = next_waypoint.road_id * 100 + next_waypoint.lane_id
                            
                        if lane["left"] == None and wp.left_lane_marking.lane_change & carla.LaneChange.Left:
                            next_waypoint = wp.get_left_lane()
                            if next_waypoint is not None and next_waypoint.lane_type == carla.LaneType.Driving and wp.road_id == next_waypoint.road_id:
                                lane["left"] = next_waypoint.road_id * 100 + next_waypoint.lane_id
                            
                for wp in wp.next(5.0):
                    if wp.road_id * 100 + wp.lane_id not in lane["next"]:
                        lane["next"].append(wp.road_id * 100 + wp.lane_id )
                            

                self.lanes[laneid] = lane

    def Load_from_File(self, filename):
        with open(filename,"rb") as f:
            self.lanes = pickle.load(f)
    def Save(self, filename):
        with open(filename,"wb") as f:
            pickle.dump(self.lanes, f)
            


        
class LaneType(Enum):
    Follow = 0
    Straight = 1
    Left = 2
    Right = 3
    