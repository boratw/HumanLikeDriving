
import glob
import os
import sys
try:
    sys.path.append(glob.glob('/home/user/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import cv2
import numpy as np
import datetime

import tensorflow.compat.v1 as tf
from lanetrace import LaneTrace
from algorithm.routepredictor_DriveStyle import RoutePredictor

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

class SafetyPotential:
    def __init__(self, laneinfo, visualize=False, record_video=False, now=None, agent_count=100):
        self.player = None
        self.visualize = visualize
        self.record_video = record_video
        self.video = None
        self.now = now

        self.cam_topview = None
        self.cam_frontview = None
        self.img_topview = None
        self.img_frontview = None

        self.agent_count = agent_count
        self.routepredictor = RoutePredictor(laneinfo, agent_count)

    def Assign_Player(self, player):
        self.player = player
        if self.visualize:
            world = player.get_world()

            bp = world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '1024')
            bp.set_attribute('image_size_y', '1024')
            self.cam_topview = world.spawn_actor(bp, carla.Transform(
                carla.Location(x=24.0, z=32.0), carla.Rotation(pitch=-90, yaw=0)), attach_to=player)
            self.cam_topview.listen(lambda image: self.on_cam_topview_update(image))

            bp = world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '1024')
            bp.set_attribute('image_size_y', '512')
            self.cam_frontview = world.spawn_actor(bp, carla.Transform(
                carla.Location(x=-7.5, z=2.5)), attach_to=player) # 2.3 1.0
            self.cam_frontview.listen(lambda image: self.on_cam_frontview_update(image))

            if self.record_video:
                if self.video:
                    self.video.release()
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                self.video = cv2.VideoWriter('recorded_' + self.now + '.avi', fourcc, 13, (2048, 1024))

    def Assign_NPCS(self, npcs):
        self.npcs = npcs
        self.routepredictor.Assign_NPCS(npcs)


    def get_target_speed(self, target_velocity_in_scenario, print_log=False, impatience=None):
        target_velocity = target_velocity_in_scenario / 3.6 # HO ADDED 20.0
        sff_potential = 0.0
        final_sff = None
        actor_distances = [999, 999, 999, 999]

        if self.player != None:
            agent_tr = self.player.get_transform()
            agent_v = self.player.get_velocity()
            agent_f = agent_tr.get_forward_vector()

            close_npcs = []
            npc_transforms = []
            npc_velocities = []
            npc_trafficlights = []
            for npci, npc in enumerate(self.npcs):
                tr = npc.get_transform()
                v = npc.get_velocity()
                npc_transforms.append(tr)
                npc_velocities.append(v)
                loc = tr.location
                if np.sqrt( (agent_tr.location.x - loc.x) ** 2 + (agent_tr.location.y - loc.y) ** 2 ) < 256: ##DISTANCE
                    close_npcs.append(npci)
                actor_distances.append(np.sqrt((agent_tr.location.x - tr.location.x) ** 2 +  (agent_tr.location.y - tr.location.y) ** 2))

                try:
                    tlight = npc.get_traffic_light()
                    tlight_state = tlight.get_state()
                except:
                    tlight_state = carla.TrafficLightState.Unknown
                npc_trafficlights.append(tlight_state)

            if len(close_npcs) > 0:
                self.routepredictor.Get_Predict_Result(close_npcs, npc_transforms, npc_velocities, agent_tr, agent_v, npc_trafficlights, impatience)

                potential = np.zeros((9, 16))


                nx = [agent_tr.location.x + agent_f.x * 5.]
                ny = [agent_tr.location.y + agent_f.y * 5.]
                for i in range(8):
                    nx.append(nx[-1] + agent_f.x * 4.05)
                    ny.append(ny[-1] + agent_f.y * 4.05)

                ni  = 0
                for npci in close_npcs:
                    f = npc_transforms[npci].get_forward_vector()
                    fx = f.x * 3.0
                    fy = f.y * 3.0
                    for i in range(self.routepredictor.output_route_num):
                        index = self.routepredictor.output_route_num * ni + i
                        prob = self.routepredictor.pred_prob[index]
                        for k in range(9):

                            dx = npc_transforms[npci].location.x - nx[k]
                            dy = npc_transforms[npci].location.y - ny[k]
                            d = (3. - np.sqrt(dx * dx + dy * dy))
                            if d > 1.:
                                d = 1.
                            if potential[k][0] < (d * prob):
                                potential[k][0] = d * prob

                            dx += fx
                            dy += fy
                            d = (3. - np.sqrt(dx * dx + dy * dy))
                            if d > 1.:
                                d = 1.
                            if potential[k][0] < (d * prob):
                                potential[k][0] = d * prob

                        for j in range(self.routepredictor.output_route_len):
                            for k in range(9):

                                dx = self.routepredictor.pred_route[index][j][0] - nx[k]
                                dy = self.routepredictor.pred_route[index][j][1] - ny[k]
                                d = (3. - np.sqrt(dx * dx + dy * dy))
                                if d > 1.:
                                    d = 1.
                                if potential[k][j + 1] < (d * prob):
                                    potential[k][j + 1] = d * prob

                                dx += fx
                                dy += fy
                                d = (3. - np.sqrt(dx * dx + dy * dy))
                                if d > 1.:
                                    d = 1.
                                if potential[k][j + 1] < (d * prob):
                                    potential[k][j + 1] = d * prob

                    ni += 1
                #print("prob field")
                #for i in range(9):
                #    print([potential[i][j] for j in range(8)])
                
                v_prob = [0.] * 13
                v_prob[1] = max([potential[0][0], potential[1][2], potential[2][4], potential[3][6]])
                v_prob[2] = max([potential[0][0], potential[1][1], potential[2][2], potential[3][3]])
                v_prob[3] = max([potential[0][0], potential[1][1] * 0.5 + potential[2][1] * 0.5, potential[3][2], potential[4][3] * 0.5 +  potential[5][3] * 0.5])
                v_prob[4] = max([potential[0][0], potential[1][0], potential[2][1], potential[4][2], potential[6][3]])
                v_prob[5] = max([potential[0][0], potential[1][0], potential[2][1] * 0.5 + potential[3][1] * 0.5,  potential[5][2], potential[7][3] * 0.5 +  potential[8][3] * 0.5])
                v_prob[6] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][1] + potential[6][2]])
                v_prob[7] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][1] * 0.5 + potential[4][1] * 0.5 + potential[7][2]])
                v_prob[8] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][1] + potential[8][2]])
                v_prob[9] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][1] * 0.5 + potential[5][1] * 0.5])
                v_prob[10] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][0], potential[5][1] ])
                v_prob[11] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][0], potential[5][1] * 0.5 + potential[6][1] * 0.5])
                v_prob[12] = max([potential[0][0], potential[1][0], potential[2][0], potential[3][0], potential[4][0], potential[5][0], potential[6][1]])
                

                #print(v_prob)
                for i in range(12, -1, -1):
                    dv = i
                    if target_velocity < dv + 0.5:
                        dv = dv + 0.5 - 2. * v_prob[i]
                        if target_velocity > dv:
                            target_velocity = dv
                    
                if target_velocity < 1: # HO ADDED
                    target_velocity = 0.

                
            if self.visualize:
                visual_output = np.zeros((1024, 2048, 3), np.uint8)
                actor_speed = np.sqrt(agent_v.x ** 2 + agent_v.y ** 2)
                if self.img_topview is not None:
                    '''                    
                    sff_visual = np.zeros((512, 512, 3), np.uint8)
                    line_visual = np.zeros((1024, 1024, 3), np.uint8)

                    my_sff_visual = np.zeros((1024, 1024, 3), np.uint8)
                    f = agent_tr.get_forward_vector()

                    expected_distance = 0
                    v = actor_speed
                    for j in range(11):
                        expected_distance += v / 5
                        if v > target_velocity:
                            v = v * 0.9 - 11.0 * 0.4
                        else:
                            v = v * 0.9 + 1.7 * 0.4
                    if expected_distance < 3:
                        expected_distance = 3

                    cv2.line(my_sff_visual, (512 - int(f.x * 12), 512 - int(f.y * 12)), 
                        (512 + int(f.x * 8 * (1.5 + expected_distance)), 512 + int(f.y * 8 * (1.5 + expected_distance))), (255, 0, 0), 20)
                    my_sff_visual = cv2.GaussianBlur(my_sff_visual, (0, 0), 11)
                    my_sff_visual = cv2.warpAffine(my_sff_visual, M, (1024, 1024))
                    my_sff_visual = my_sff_visual[64:576, 256:768]
                    my_sff_visual = cv2.resize(my_sff_visual, (1024, 1024), interpolation=cv2.INTER_LINEAR)

                    #cv2.polylines(line_visual, route_line, False, (0, 255, 0), 2)

                    bb = self.player.bounding_box.get_world_vertices(self.player.get_transform())
                    bb_list = [[locx + int(bb[0].x * 8), locy + int(bb[0].y * 8)], [locx + int(bb[2].x * 8), locy + int(bb[2].y * 8)], 
                            [locx + int(bb[6].x * 8), locy + int(bb[6].y * 8)], [locx + int(bb[4].x * 8), locy + int(bb[4].y * 8)]]
                    cv2.polylines(line_visual, np.array([bb_list], dtype=np.int32), True, (255, 0, 0), 2)
                    for npci, npc in enumerate(self.npcs):
                        bb = npc.bounding_box.get_world_vertices(npc.get_transform())
                        bb_list = [[locx + int(bb[0].x * 8), locy + int(bb[0].y * 8)], [locx + int(bb[2].x * 8), locy + int(bb[2].y * 8)], 
                                [locx + int(bb[6].x * 8), locy + int(bb[6].y * 8)], [locx + int(bb[4].x * 8), locy + int(bb[4].y * 8)]]
                        cv2.polylines(line_visual, np.array([bb_list], dtype=np.int32), True, (0, 0, 255), 2)
                    
                    if final_sff is not None:
                        sff_visual[:, :, 2] = final_sff
                    sff_visual = cv2.resize(sff_visual, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    line_visual = cv2.warpAffine(line_visual, M, (1024, 1024))
                    line_visual = line_visual[64:576, 256:768]
                    line_visual = cv2.resize(line_visual, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    mask = np.mean(line_visual, axis=2, dtype=np.uint8)

                    final_visual = cv2.addWeighted(self.img_topview, 0.5, sff_visual, 1.0, 0)
                    final_visual = cv2.add(final_visual, my_sff_visual)
                    cv2.copyTo(line_visual, mask, final_visual)
                    visual_output[:, :1024] = final_visual
                    '''
                    visual_output[:, :1024] = self.img_topview
                if self.img_frontview is not None:
                    visual_output[:512, 1024:] = self.img_frontview
                cv2.putText(visual_output, "Current Speed : %dkm/h" % int(actor_speed * 3.6), (1050, 600), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (255, 255, 255), thickness=7)
                cv2.putText(visual_output, "Target Speed : %dkm/h" % int(round(target_velocity * 3.6)), (1050, 790), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (255, 255, 255), thickness=7)
                cv2.putText(visual_output, "Safety Potential : %.3f" % sff_potential, (1050, 980), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (255, 255, 255), thickness=7)
                cv2.imshow("visual_output", visual_output)

                if self.record_video:
                    self.video.write(visual_output)

            actor_distances.sort()
            cv2.waitKey(1)

        if target_velocity < 0.:
            target_velocity = 0.

        actor_speed = np.sqrt(agent_v.x ** 2 + agent_v.y ** 2) * 3.6
        sff_log = str(actor_speed) + "\t" + str(round(target_velocity * 3.6)) + "\t" + "\t".join([str(v_prob[j]) for j in range(13)])
        if print_log:
            return target_velocity, sff_log
        else:
            return target_velocity
        #print(target_velocity)
            #cv2.imshow("SafetyPotential2", final2)
            #cv2.waitKey(1)

    def on_cam_topview_update(self, image):
        if not image:
            return

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_image = np.reshape(image_data, (image.height, image.width, 4))
        np_image = np_image[:, :, :3]
        np_image = np_image[:, :, ::-1]
        #np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        #self.img_topview = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
        self.img_topview = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    def on_cam_frontview_update(self, image):
        if not image:
            return

        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_image = np.reshape(image_data, (image.height, image.width, 4))
        np_image = np_image[:, :, :3]
        np_image = np_image[:, :, ::-1]
        self.img_frontview = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

    def destroy(self):
        if self.visualize:
            if self.cam_topview:
                self.cam_topview.stop()
                self.cam_topview.destroy()
            if self.cam_frontview:
                self.cam_frontview.stop()
                self.cam_frontview.destroy()
            if self.record_video:
                self.video.release()