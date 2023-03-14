
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
            
class RouteTracer(object):
    def __init__(self, laneinfo):
        self.laneinfo = laneinfo
        self.laneid = None
        self.laneindex = 0

    def Trace(self, x, y, yaw):
        if self.laneid == None:
            self.Find(x, y, yaw)
        return self.GenRoute(x, y, yaw)

    def GenRoute(self, x, y, yaw):
        if self.laneid != None:
            poslist = self.laneinfo.lanes[self.laneid]["pos"]
            if self.laneindex < len(poslist[self.laneindex]) - 1:
                dist1 = (poslist[self.laneindex][0] - x) * (poslist[self.laneindex][0] - x) \
                    + (poslist[self.laneindex][1] - y) * (poslist[self.laneindex][1] - y)
                dist2 = (poslist[self.laneindex + 1][0] - x) * (poslist[self.laneindex + 1][0] - x) \
                    + (poslist[self.laneindex + 1][1] - y) * (poslist[self.laneindex + 1][1] - y)
                if dist1 > dist2:
                    self.laneindex += 1
            else:
                dist1 = (poslist[self.laneindex][0] - x) * (poslist[self.laneindex][0] - x) \
                    + (poslist[self.laneindex][1] - y) * (poslist[self.laneindex][1] - y)
                if dist1 > 9:
                    self.FindFromList(x, y, yaw, self.laneinfo.lanes[self.laneid]["next"])
                    return self.GenRoute(x, y, yaw)

            d1 = None
            d2 = None
            d3 = None
            for id in self.laneinfo.lanes[self.laneid]["next"]:
                if self.laneinfo.lanes[id]["type"] == LaneType.Left:
                    d2 = id
                elif self.laneinfo.lanes[id]["type"] == LaneType.Right:
                    d3 = id
                else:
                    d1 = id
            if d1 == None:
                if d2 == None:
                    if d3 != None:
                        d1 = d3
                else:
                    d1 = d2
            d1, nextlineused = self.Follow(x, y, yaw, d1)
            if nextlineused and d2 != None:
                d2, _ = self.Follow(x, y, yaw, d2)
            else:
                d2 = d1
            if nextlineused and d3 != None:
                d3, _ = self.Follow(x, y, yaw, d3)
            else:
                d3 = d1
                
            return [ d1, d2, d3 ]
        
        return None


    def Find(self, x, y, yaw):
        minlane = None
        mindist = 9
        minindex = 0
        for laneid, lane in self.laneinfo.lanes.items():
            yawdiff = yaw - lane["orientation"]
            if yawdiff < 0:
                yawdiff = -yawdiff
            if yawdiff < np.pi * 0.25 or yawdiff > np.pi * 1.75:
                poslist = lane["pos"]
                if poslist[0][0] > poslist[-1][0]:
                    if x < poslist[-1][0] - 3 or x > poslist[0][0] + 3:
                        continue
                else:
                    if x < poslist[0][0] - 3 or x > poslist[-1][0] + 3:
                        continue

                if poslist[0][1] > poslist[-1][1]:
                    if y < poslist[-1][1] - 3 or y > poslist[0][1] + 3:
                        continue
                else:
                    if y < poslist[0][1] - 3 or y > poslist[-1][1] + 3:
                        continue

                for idx, pos in enumerate(poslist):
                    dist = (pos[0] - x) * (pos[0] - x) + (pos[1] - y) * (pos[1] - y)
                    if dist < mindist:
                        minlane = laneid
                        minindex = idx
                        mindist = dist
                
        self.laneid = minlane
        self.laneindex = minindex

    def FindFromList(self, x, y, yaw, lanelist):
        minlane = None
        mindist = 999999
        minindex = 0
        for laneid in lanelist:
            lane = self.laneinfo.lanes[laneid]
            yawdiff = yaw - lane["orientation"]
            if yawdiff < 0:
                yawdiff = -yawdiff
            if yawdiff < np.pi * 0.25 or yawdiff > np.pi * 1.75:
                poslist = lane["pos"]
                if poslist[0][0] > poslist[-1][0]:
                    if x < poslist[-1][0] - 3 or x > poslist[0][0] + 3:
                        continue
                else:
                    if x < poslist[0][0] - 3 or x > poslist[-1][0] + 3:
                        continue

                if poslist[0][1] > poslist[-1][1]:
                    if y < poslist[-1][1] - 3 or y > poslist[0][1] + 3:
                        continue
                else:
                    if y < poslist[0][1] - 3 or y > poslist[-1][1] + 3:
                        continue

                for idx, pos in enumerate(poslist):
                    dist = (pos[0] - x) * (pos[0] - x) + (pos[1] - y) * (pos[1] - y)
                    if dist < mindist:
                        minlane = laneid
                        minindex = idx
        self.laneid = minlane
        self.laneindex = minindex

    def Follow(self, x, y, yaw, nextlaneid):
        poslist = self.laneinfo.lanes[self.laneid]["pos"]
        posindex = self.laneindex
        laneid = None

        curx = 0
        cury = 0

        rx = 0
        ry = 0
        rremain = 0
        rposx = x
        rposy = y

        res = []
        for d in range(5):
            dremain = 5
            while dremain > rremain:
                curx = poslist[posindex][0]
                cury = poslist[posindex][1]
                dremain -= rremain
                posindex += 1
                if posindex == len(poslist):
                    if laneid == None:
                        laneid = nextlaneid
                    else:
                        laneid = self.laneinfo.lanes[laneid]["next"][0]
                    poslist = self.laneinfo.lanes[laneid]["pos"]
                    posindex = 0
                rx = poslist[posindex][0] - rposx
                ry = poslist[posindex][1] - rposy
                rposx = poslist[posindex][0]
                rposy = poslist[posindex][1]
                rremain = np.sqrt(rx * rx + ry * ry)
                if dremain <= rremain:
                    rx /= rremain
                    ry /= rremain
            curx += rx * dremain
            cury += ry * dremain
            rremain -= dremain
            nx, ny = rotate(curx - x, cury - y, yaw)
            res.extend([curx - x, cury - y])
        return res, (laneid != None)
            
    def Reset(self):
        self.laneid = None
        self.laneindex = 0




        
class LaneType(Enum):
    Follow = 0
    Straight = 1
    Left = 2
    Right = 3
    
def rotate(posx, posy, yaw):
    return posx * np.sin(yaw) + posy * np.cos(yaw), posx * np.cos(yaw) - posy * np.sin(yaw)
