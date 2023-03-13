
import carla
import numpy as np
import random
from enum import Enum
from collections import deque

class LaneInfo(object):
    def __init__(self, world):
        self.map = world.get_map()

        waypoints = self.map.generate_waypoints(2.0)
        self.lanes = {}

        for w in waypoints:
            laneid = w.road_id * 100 + w.lane_id
            if laneid not in self.lanes:
                lane = {"list" : [w], "left" : None, "right" : None, "type" : LaneType.Follow, "next" : []}
                for wp in w.previous_until_lane_start(2.0):
                    if wp.road_id == w.road_id:
                        lane["list"].append(wp)
                lane["list"] = lane["list"][::-1]
                for wp in w.next_until_lane_end(2.0):
                    if wp.road_id == w.road_id:
                        lane["list"].append(wp)

                if wp.is_junction == True:
                    turning = np.sin((lane["list"][-1].transform.rotation.yaw - lane["list"][0].transform.rotation.yaw) * 0.017453293)
                    if turning < -0.5:
                        lane["type"] = LaneType.Left
                    elif turning > 0.5:
                        lane["type"] = LaneType.Right
                    else:
                        lane["type"] = LaneType.Straight
                else:
                    for wp in lane["list"]:
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

class RoutePlanner(object):
    def __init__(self, laneinfo):
        self.laneinfo = laneinfo
        self.route = deque(maxlen = 100)
        self.routelaneid = deque(maxlen = 100)
        self.lastlane = None
        self.lastlaneid = 0
        self.lastindex = 0
        self.curlaneid = None

        self.changable_left = False
        self.changable_right = False

    def Move(self, loc):
        while len(self.route) > 0:
            if self.route[0].location.distance(loc) < 3.0:
                self.route.popleft()
                self.routelaneid.popleft()
            else:
                break
        
        if len(self.route) == 0:
            if self.New(loc) == False:
                return
        
        while len(self.route) < 20:
            self.Next()

        if self.curlaneid != self.routelaneid[0]:
            self.curlaneid = self.routelaneid[0]
            self.changable_left = (self.laneinfo.lanes[self.curlaneid]["left"] != None)
            self.changable_right = (self.laneinfo.lanes[self.curlaneid]["right"] != None)

    def New(self, loc):
        self.route.clear()
        self.routelaneid.clear()

        w = self.laneinfo.map.get_waypoint(loc)
        if w == None:
            return False
        laneid = w.road_id * 100 + w.lane_id
        if laneid not in self.laneinfo.lanes:
            return False
        lane = self.laneinfo.lanes[laneid]
        for i in range(len(lane["list"]) - 1, -1, -1):
            if lane["list"][i].transform.location.distance(w.transform.location) < 3.0:
                break

        self.route.append(lane["list"][i].transform)
        self.routelaneid.append(laneid)
        self.lastlane = lane
        self.lastlaneid = laneid
        self.lastindex = i+1
        self.curlaneid = None
        return True
    
    def Next(self):
        
        if self.lastindex == len(self.lastlane["list"]):
            laneid = random.choice(self.lastlane["next"])
            self.lastlane = self.laneinfo.lanes[laneid]
            self.lastlaneid = laneid
            self.lastindex = 0
        self.route.append(self.lastlane["list"][self.lastindex].transform)
        self.routelaneid.append(self.lastlaneid)
        self.lastindex += 1

    def Clear(self):
        self.route.clear()
        self.routelaneid.clear()


        
class LaneType(Enum):
    Follow = 0
    Straight = 1
    Left = 2
    Right = 3
    