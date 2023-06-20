
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

from network.DrivingStyle8 import DrivingStyleLearner
import tensorflow.compat.v1 as tf
from lanetrace import LaneTrace

state_len = 53 
nextstate_len = 10
route_len = 20
action_len = 3
agent_for_each_train = 8
global_latent_len = 4
l2_regularizer_weight = 0.0001
global_regularizer_weight = 0.001

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

        tf.disable_eager_execution()
        self.sess = tf.Session()
        with self.sess.as_default():
            self.learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, global_latent_len=global_latent_len, 
                                            l2_regularizer_weight=l2_regularizer_weight, global_regularizer_weight=global_regularizer_weight, route_len=route_len, action_len= action_len)
            learner_saver = tf.train.Saver(var_list=self.learner.trainable_dict, max_to_keep=0)
            learner_saver.restore(self.sess, "train_log/DrivingStyle8_fake/log_16-06-2023-18-06-20_150.ckpt")

        self.lane_tracers = [LaneTrace(laneinfo, 10) for _ in range(agent_count)]
        self.agent_count = agent_count

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
        self.agent_count = len(npcs)
        self.npc_transforms = []
        self.npc_velocities = []

    def Get_Predict_Result(self):
        state_dic = []
        route_dic = []
        action_dic = []
        for i in self.close_npcs:
            tr = self.npc_transforms[i]
            v = self.npc_velocities[i]
            x = tr.location.x
            y = tr.location.y
            yawsin = np.sin(tr.rotation.yaw  * -0.017453293)
            yawcos = np.cos(tr.rotation.yaw  * -0.017453293)
            other_vcs = []
            for j in range(self.agent_count):
                if i != j:
                    relposx = self.npc_transforms[j].location.x - x
                    relposy = self.npc_transforms[j].location.y - y
                    px, py = rotate(relposx, relposy, yawsin, yawcos)
                    vx, vy = rotate(self.npc_velocities[j].x, self.npc_velocities[j].y, yawsin, yawcos)
                    relyaw = (self.npc_transforms[j].rotation.yaw - tr.rotation.yaw)   * 0.017453293
                    other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
            other_vcs = np.array(sorted(other_vcs, key=lambda s: s[6]))
            velocity = np.sqrt(v.x ** 2 + v.y ** 2)

            traced, tracec = self.lane_tracers[i].Trace(x, y)
            route = []
            if traced == None:
                for trace in range(action_len):
                    waypoints = []
                    for j in range(route_len // 2):
                        waypoints.extend([0., 0.])
                    route.append(waypoints)
            else:
                for trace in traced:
                    waypoints = []
                    for j in trace:
                        px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                        waypoints.extend([px, py])
                    route.append(waypoints)

            state = np.concatenate([[velocity, 0., px, py, 0.], other_vcs[:8,:6].flatten()])
            state_dic.append(state)
            state_dic.append(state)
            state_dic.append(state)
            route_dic.append(route)
            route_dic.append(route)
            route_dic.append(route)
            action_dic.extend([0, 1, 2])

        global_latent_dic = [[-1.4275596704333653, 2.429559433754484, 3.353914895291514, 1.9095644036140866] for _ in range(len(state_dic)) ]
        with self.sess.as_default():
            self.pred_prob = self.learner.get_decoded_action(state_dic, route_dic, global_latent_dic)
            self.pred_route = self.learner.get_decoded_route(state_dic, route_dic, action_dic, global_latent_dic)


    def get_target_speed(self, target_velocity_in_scenario, route, visualize=False, print_log=False):
        target_velocity = target_velocity_in_scenario # HO ADDED 20.0
        sff_potential = 0.0
        final_sff = None
        actor_distances = [999, 999, 999, 999]

        if self.player != None:
            agent_tr = self.player.get_transform()
            agent_v = self.player.get_velocity()
            M = cv2.getRotationMatrix2D((512, 512), agent_tr.rotation.yaw + 90, 1.0)

            locx = 512 - int(agent_tr.location.x * 8)
            locy = 512 - int(agent_tr.location.y * 8)
            loctr = np.array([locx, locy], np.int32)

            self.close_npcs = []
            self.npc_transforms = []
            self.npc_velocities = []
            for npci, npc in enumerate(self.npcs):
                tr = npc.get_transform()
                v = npc.get_velocity()
                self.npc_transforms.append(tr)
                self.npc_velocities.append(v)
                loc = tr.location
                front_loc = loc + tr.get_forward_vector() * 5.
                if np.sqrt( (agent_tr.location.x - loc.x) ** 2 + (agent_tr.location.y - loc.y) ** 2 ) < 256: ##DISTANCE
                    self.close_npcs.append(npci)
                actor_distances.append(np.sqrt((agent_tr.location.x - tr.location.x) ** 2 +  (agent_tr.location.y - tr.location.y) ** 2))

            if len(self.close_npcs) > 0:
                self.Get_Predict_Result()

                screen = np.zeros((1024, 1024), np.uint8)
                line_screen = np.zeros((1024, 1024), np.uint8)


                route_line = [[512, 512]]
                for i, (waypoint, roadoption) in enumerate(route):
                    route_line.append([locx + int(waypoint.transform.location.x * 8), locy + int(waypoint.transform.location.y * 8)])
                    if i == 20:
                        break
                route_line = np.array([route_line], dtype=np.int32)
                cv2.polylines(line_screen, route_line, False, (255,), 20)

                #vx_array = -record[step:step+50, :, 3] * sin_array + record[step:step+50, :, 4] * cos_array
                #vy_array = -record[step:step+50, :, 3] * cos_array - record[step:step+50, :, 4] * sin_array


                new_screen = np.zeros((4, 1024, 1024), np.uint8)
                ni  = 0
                for npci in self.close_npcs:
                    yawsin = np.sin(self.npc_transforms[npci].rotation.yaw * 0.017453293)
                    yawcos = np.cos(self.npc_transforms[npci].rotation.yaw * 0.017453293)
                

                    for i in range(3):
                        line = []
                        x, y = locx + self.npc_transforms[npci].location.x * 8, locy + self.npc_transforms[npci].location.y * 8
                        line.append([x, y])
                        for j in range(5):
                            px, py = rotate(self.pred_route[3 * ni + i][2 * j], self.pred_route[3 * ni + i][2 * j + 1], yawsin, yawcos)
                            x += px * 8
                            y += py * 8
                            line.append([x, y])

                        color = int(self.pred_prob[3 * ni + i][i] * 256)
                        cv2.polylines(new_screen[i], np.array([line], dtype=np.int32), False, (color,), 20)
                    ni += 1
                for i in range(4):
                    blurred1 = cv2.GaussianBlur(new_screen[i], (0, 0), 11)
                    screen = cv2.add(screen, blurred1)

                final_sff = cv2.warpAffine(screen, M, (1024,1024))
                final_line = cv2.warpAffine(line_screen, M, (1024,1024))
                final_sff = final_sff[64:576, 256:768]
                final_line = final_line[64:576, 256:768]
                #cv2.imshow("final", final)
                #cv2.imshow("final_line", final_line)

                final = cv2.resize(final_sff[192:448, 128:384], (64, 64), interpolation=cv2.INTER_AREA)
                final_line = cv2.resize(final_line[192:448, 128:384], (64, 64), interpolation=cv2.INTER_AREA)

                #final2 = final[48:108, 118:138]
                #cv2.imshow("SafetyPotential", final2)
                #cv2.waitKey(1)
                final_mean = np.clip(np.max(final_line.astype(np.float32) * final.astype(np.float32) / 25600., axis=1), 0., 1.)
                sff_potential = np.mean(final_mean) 

                for i in range(60):
                    new_velocity = 20. - 0.35 * i
                    target_velocity = target_velocity * (1 - final_mean[i]) + new_velocity * final_mean[i]

                if target_velocity < 1: # HO ADDED
                    target_velocity = 0.

                if target_velocity < 0.:
                    target_velocity = 0.
                
            if self.visualize:
                visual_output = np.zeros((1024, 2048, 3), np.uint8)
                actor_speed = np.sqrt(agent_v.x ** 2 + agent_v.y ** 2)
                if self.img_topview is not None:
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

        sff_log = str(round(target_velocity * 3.6)) + "\t" + str(sff_potential) + "\t" + str(actor_distances[0]) + "\t" + str(actor_distances[1])+ "\t" + str(actor_distances[2])+ "\t" + str(actor_distances[3])
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