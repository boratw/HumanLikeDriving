
import numpy as np
import cv2
'''
        distance_to_leading_vehicle = [ np.random.uniform(5.0, 15.0) for i in range(agent_num) ]
        vehicle_lane_offset = [ np.random.uniform(-0.3, 0.3) for i in range(agent_num) ]
        vehicle_speed = [ np.random.uniform(-50.0, 50.0) for i in range(agent_num) ]
        impatient_lane_change = [ np.abs(np.random.normal(0.0, 100.0) + 20.) for i in range(agent_num) ]
        vel_disp = [ np.random.uniform(0.1, 1.0) for i in range(agent_num) ]
        lane_disp = [ np.random.uniform(0.1, 1.0) for i in range(agent_num) ]

        
        for x in range(agent_count):
            for t in range(6):
                log_txt.write(str(param_vectors[x][t]) + "\t")
            log_txt.write(str(prob_res[x] / res_num) + "\t")
            log_txt.write(str(prob_var[x] / res_num - (prob_res[x] / res_num) ** 2))
            for t in range(6):
                log_txt.write("\t" + str(close_route_res[x][t] / res_num))
            for t in range(6):
                log_txt.write("\t" + str(close_route_var[x][t] / res_num - (close_route_res[x][t] / res_num) ** 2))
            for t in range(6):
                log_txt.write("\t" + str(maximum_route_res[x][t] / res_num))
            for t in range(6):
                log_txt.write("\t" + str(maximum_route_var[x][t] / res_num - (maximum_route_res[x][t] / res_num) ** 2))
            log_txt.write("\n")
'''
def GetColor(r):
    if r < 0.33333333:
        return (0, 0, int(r * 768))
    elif r < 0.66666666:
        return (0, int((r - 0.33333333) * 768), 255)
    else:
        return (0, 255, int((1. - r) * 768))

log_file = open("test_log/log.txt", "wt")
for module_n in range(7):
    res = [0.] * 32
    res_num = 0

    lo_vs = np.zeros((20, 20))
    lo_vs_num = np.zeros((20, 20)) + 1e-7
    lo_il = np.zeros((20, 20))
    lo_il_num = np.zeros((20, 20)) + 1e-7
    vd_vs = np.zeros((20, 20))
    vd_vs_num = np.zeros((20, 20)) + 1e-7



    with open("test_log/module" + str(module_n) + ".txt", "rt") as f:
        for line in f.readlines():
            s = line.split("\t")
            if len(s) == 32:
                v = [float(t) for t in s]
                for t in range(26):
                    res[t] += v[t + 6]
                score = 0.
                for t in range(6):
                    r = np.exp(-((v[t + 8] / 10.) ** 2 )) * v[6]
                    res[t + 26] += r
                    score += r * (2 ** (t // 3))
                score /= 14
                res_num += 1

                lo_index = int(np.clip((v[1] + 0.3) * 16 / 0.6, 0, 15)) + 1
                vs_index = int(np.clip((v[2] + 50.) * 16 / 100., 0, 15)) + 1
                il_index = int(np.clip((v[3] - 20.) * 16 / 150, 0, 15)) + 1
                vd_index = int(np.clip((v[5] - 0.1) * 16 / 0.9, 0, 15)) + 1

                lo_vs[lo_index - 1 : lo_index + 1, vs_index - 1 : vs_index + 1] += score
                #lo_vs_num[lo_index - 1 : lo_index + 1, vs_index - 1 : vs_index + 1] += 1
                lo_il[lo_index - 1 : lo_index + 1, il_index - 1 : il_index + 1] += score
                #lo_il_num[lo_index - 1 : lo_index + 1, il_index - 1 : il_index + 1] += 1
                vd_vs[vs_index - 1 : vs_index + 1, vd_index - 1 : vd_index + 1] += score
                #vs_vd_num[vs_index - 1 : vs_index + 1, vd_index - 1 : vd_index + 1] += 1

        lo_vs /= np.max(lo_vs)
        lo_il /= np.max(lo_il)
        vd_vs /= np.max(vd_vs)

        log_file.write(str(module_n) + "\t" + "\t".join([str(r / res_num) for r in res]) + "\n")
        for name, t, t_num in [("lo_vs", lo_vs, lo_vs_num), ("lo_il", lo_il, lo_il_num), ("vs_vd", vd_vs, vd_vs_num)]:
            res_img = np.zeros((256, 256, 3), np.uint8)
            for x in range(2, 18):
                for y in range(2, 18):
                    cv2.rectangle(res_img, (x * 16 - 32, y * 16 - 32), (x * 16 - 16, y * 16 - 16), GetColor(t[x, y]), -1)
            cv2.imwrite("test_log/log_" + str(module_n) + "_" + name + ".png", res_img)

