
import carla
import numpy as np
import random
from enum import Enum
from collections import deque
import pickle
from laneinfo import LaneType

            
class LaneTrace(object):
    def __init__(self, laneinfo, trace_length = 10):
        self.laneinfo = laneinfo
        self.laneid = None
        self.laneindex = 0
        self.trace_length = trace_length
        self.lanechanged = False

    def Find(self, x, y):
        if self.laneid != None:
            idx = self.FindFromLane(x, y, self.laneid)
            if idx != -1:
                self.laneindex = idx
                self.lanechanged = False
                return
            for laneid in self.laneinfo.lanes[self.laneid]["next"]:
                idx = self.FindFromLane(x, y, laneid)
                if idx != -1:
                    self.laneid = laneid
                    self.laneindex = idx
                    self.lanechanged = False
                    return
            if self.laneinfo.lanes[self.laneid]["left"] != None:
                idx = self.FindFromLane(x, y, self.laneinfo.lanes[self.laneid]["left"])
                if idx != -1:
                    self.laneid = self.laneinfo.lanes[self.laneid]["left"]
                    self.laneindex = idx
                    self.lanechanged = True
                    return
            if self.laneinfo.lanes[self.laneid]["right"] != None:
                idx = self.FindFromLane(x, y, self.laneinfo.lanes[self.laneid]["right"])
                if idx != -1:
                    self.laneid = self.laneinfo.lanes[self.laneid]["right"]
                    self.laneindex = idx
                    self.lanechanged = True
                    return


        for laneid, lane in self.laneinfo.lanes.items():
            idx = self.FindFromLane(x, y, laneid)
            if idx != -1:
                self.laneid = laneid
                self.laneindex = idx
                self.lanechanged = False
                return


    def FindFromLane(self, x, y, laneid):
        poslist = self.laneinfo.lanes[laneid]["pos"]
        mindist = 9
        minindex = -1
        for idx, pos in enumerate(poslist):
            dist = (pos[0] - x) * (pos[0] - x) + (pos[1] - y) * (pos[1] - y)
            if dist < mindist:
                mindist = dist
                minindex = idx
        return minindex
    
    def FindMinimumFromLane(self, x, y, laneid):
        poslist = self.laneinfo.lanes[laneid]["pos"]
        mindist = 999999
        minindex = -1
        for idx, pos in enumerate(poslist):
            dist = (pos[0] - x) * (pos[0] - x) + (pos[1] - y) * (pos[1] - y)
            if dist < mindist:
                mindist = dist
                minindex = idx
        return minindex


    def Trace(self, x, y):
        self.Find(x, y)
        output = [[], [], []]
        if self.laneid != None:
            l1 = self.laneinfo.lanes[self.laneid]
            l2 = self.laneinfo.lanes[l1["left"]] if l1["left"] != None else None
            l3 = self.laneinfo.lanes[l1["right"]] if l1["right"] != None else None
            i1 = self.laneindex
            i2 = self.laneindex
            i3 = self.laneindex
            for i in range(self.trace_length):
                if l1 != None:
                    if i1 >= len(l1["pos"]):
                        if len(l1["next"]) > 0:
                            l1 = self.laneinfo.lanes[l1["next"][0]]
                            i1 = 1
                        else:
                            l1 = None
                            i1 = 1

                if l1 != None:
                    output[0].append([l1["pos"][i1][0], l1["pos"][i1][1]])
                    i1 += 2
                else:
                    if len(output[0]) > 0:
                        output[0].append(output[0][-1])
                    else:
                        output[0].append([x, y])

                if l2 != None:
                    if i2 >= len(l2["pos"]):
                        if len(l2["next"]) > 0:
                            l2 = self.laneinfo.lanes[l2["next"][0]]
                            i2 = 1
                        else:
                            l2 = None
                            i2 = 1

                if l2 != None:
                    output[1].append([l2["pos"][i2][0], l2["pos"][i2][1]])
                    i2 += 2
                else:
                    if len(output[1]) > 0:
                        output[1].append(output[1][-1])
                    else:
                        output[1].append([x, y])


                if l3 != None:
                    if i3 >= len(l3["pos"]):
                        if len(l3["next"]) > 0:
                            l3 = self.laneinfo.lanes[l3["next"][0]]
                            i3 = 1
                        else:
                            l3 = None
                            i3 = 1

                if l3 != None:
                    output[2].append([l3["pos"][i3][0], l3["pos"][i3][1]])
                    i3 += 2
                else:
                    if len(output[2]) > 0:
                        output[2].append(output[2][-1])
                    else:
                        output[2].append([x, y])


            return output, [l != None for l in [l1, l2, l3]]
        else:
            return None, [False for _ in range(3)]



                    
