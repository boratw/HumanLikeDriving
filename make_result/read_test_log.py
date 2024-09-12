
import numpy as np
import cv2
import os

def GetColor(r):
    if r < 0.33333333:
        return (0, 0, int(r * 768))
    elif r < 0.66666666:
        return (0, int((r - 0.33333333) * 768), 255)
    else:
        return (0, 255, int((1. - r) * 768))

for module_n in range(3):


    lo = np.zeros((102))
    il = np.zeros((102))
    vd = np.zeros((102))
    vs = np.zeros((102))

    lo_num = np.zeros((102)) + 1e-7
    il_num = np.zeros((102)) + 1e-7
    vd_num = np.zeros((102)) + 1e-7
    vs_num = np.zeros((102)) + 1e-7

    lo_il = np.zeros((18, 18))
    lo_il_num = np.zeros((18, 18)) + 1e-7
    vd_vs = np.zeros((18, 18))
    vd_vs_num = np.zeros((18, 18)) + 1e-7

    exp_names = os.listdir("test_log/log4_5")
    for exp_name in exp_names:
        if exp_name.startswith("module" + str(module_n) + "_data"):

            with open("test_log/log4_5/" + exp_name, "rt") as f:
                for line in f.readlines():
                    s = line.split("\t")
                    if len(s) == 32:
                        v = [float(t) for t in s]
                        score = 0.
                        for t in range(1):
                            r = np.exp(-((v[t + 8] * 5.) ** 2 )) * v[6]
                            score += r * (2 ** (-(t // 2)))
                        #distance_to_leading_vehicle = [ np.random.uniform(-5.0, 5.0) + 10.  for i in range(agent_num) ]
                        #vehicle_lane_offset = [ np.random.uniform(-1.0, 1.0) * (lane_offset_mag / 8) for i in range(agent_num) ]
                        #vehicle_speed = [ np.random.uniform(-50.0, 50.0) * (vel_ratio_mag / 8) for i in range(agent_num) ]
                        #impatient_lane_change = [ np.random.uniform(200., 1200.)  for i in range(agent_num) ]
                        vd_index = int(np.clip((v[0] - 5.0) * 16 / 10., 0, 15)) + 1
                        lo_index = int(np.clip((v[1] + 1.0) * 16 / 2., 0, 15)) + 1
                        vs_index = int(np.clip((v[2] + 50.) * 16 / 100., 0, 15)) + 1
                        il_index = int(np.clip((v[3] - 200.) * 16 / 1000, 0, 15)) + 1

                        lo_il[lo_index - 1 : lo_index + 1, il_index - 1 : il_index + 1] += score
                        lo_il_num[lo_index - 1 : lo_index + 1, il_index - 1 : il_index + 1] += 1
                        vd_vs[vs_index - 1 : vs_index + 1, vd_index - 1 : vd_index + 1] += score
                        vd_vs_num[vs_index - 1 : vs_index + 1, vd_index - 1 : vd_index + 1] += 1

                        vd_index = int(np.clip((v[0] - 5.0) * 100 / 10., 0, 99)) + 1
                        lo_index = int(np.clip((v[1] + 1.0) * 100 / 2., 0, 99)) + 1
                        vs_index = int(np.clip((v[2] + 50.) * 100 / 100., 0, 99)) + 1
                        il_index = int(np.clip((v[3] - 200.) * 100 / 1000, 0, 99)) + 1

                        vd[vd_index - 1 : vd_index + 1] += score
                        lo[lo_index - 1 : lo_index + 1] += score
                        vs[vs_index - 1 : vs_index + 1] += score
                        il[il_index - 1 : il_index + 1] += score
                        vd_num[vd_index - 1 : vd_index + 1] += 1
                        lo_num[lo_index - 1 : lo_index + 1] += 1
                        vs_num[vs_index - 1 : vs_index + 1] += 1
                        il_num[il_index - 1 : il_index + 1] += 1


    lo_il /= lo_il_num
    vd_vs /= vd_vs_num
    #lo_il /= 5
    #vd_vs /= 5

    lo /= lo_num
    vs /= vs_num
    il /= il_num
    vd /= vd_num

    for name, t, t_num in [("lo_il", lo_il, lo_il_num), ("vs_vd", vd_vs, vd_vs_num)]:
        res_img = np.zeros((256, 256, 3), np.uint8)
        for x in range(2, 18):
            for y in range(2, 18):
                cv2.rectangle(res_img, (x * 16 - 32, y * 16 - 32), (x * 16 - 16, y * 16 - 16), GetColor(t[x, y]), -1)
        cv2.imwrite("test_log/log4_5/parse/log_" + str(module_n) + "_" + name + ".png", res_img)



    for name, t, t_num in [("lo", lo, lo_num), ("vd", vd, vd_num), ("vs", vs, vs_num), ("il", il, il)]:
        with open("test_log/log4_5/parse/log_" + str(module_n) + "_" + name + ".txt", "wt") as w:
            for x in range(1, 101):
                w.write(str(t[x]) + "\n")
            