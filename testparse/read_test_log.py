
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

for module_n in range(7):
    res = [0.] * 32
    res_num = 0

    index_sum = np.zeros((8, 20, 32))
    index_num = np.zeros((8, 20, 32))


    with open("test_log/module" + str(module_n) + ".txt", "rt") as f:
        for line in f.readlines():
            s = line.split("\t")
            if len(s) == 32:
                v = [float(t) for t in s]
                '''
                lo_index = int(np.clip((v[1] + 0.3) * 16 / 0.6, 0, 15)) + 1
                vs_index = int(np.clip((v[2] + 50.) * 16 / 100., 0, 15)) + 1
                il_index = int(np.clip((v[3] - 20.) * 16 / 150, 0, 15)) + 1
                vd_index = int(np.clip((v[5] - 0.1) * 16 / 0.9, 0, 15)) + 1
                '''

                vs_index = int(np.clip((v[2] + 40.) / 80., 0, 0.9999) * 20)
                lo_index = int(np.clip((v[1] + 0.3) / 0.6, 0, 0.9999) * 20)
                il_index = int(np.clip((v[3] - 20.) / 100, 0, 0.9999) * 20)
                vd_index = int(np.clip((v[4] - 0.5) / 0.5, 0, 0.9999) * 20)
                ld_index = int(np.clip((v[5] - 0.1) / 0.9, 0, 0.9999) * 20)
                vs_lo_index = int(np.clip((v[2] + 50.) / 100., 0, 1) * np.clip((v[1] + 0.3) / 0.6, 0, 1) * 20)
                vs_lo_vd_index = int(np.clip((v[2] + 50.) / 100., 0, 1) * np.clip((v[1] + 0.3) / 0.6, 0, 1) * np.clip((v[5] - 0.1) / 0.9, 0, 15) * 20)
                vd_ld_index = int(np.clip((v[5] - 0.1) / 0.9, 0, 15) * np.clip((v[5] - 0.1) / 0.9, 0, 0.9999) * 20)

                index_sum[0][vs_index] += np.array(v)
                index_num[0][vs_index] += 1

                index_sum[1][lo_index] += np.array(v)
                index_num[1][lo_index] += 1

                index_sum[2][vd_index] += np.array(v)
                index_num[2][vd_index] += 1

                index_sum[3][il_index] += np.array(v)
                index_num[3][il_index] += 1

                index_sum[4][ld_index] += np.array(v)
                index_num[4][ld_index] += 1

                index_sum[5][vs_lo_index] += np.array(v)
                index_num[5][vs_lo_index] += 1

                index_sum[6][vs_lo_vd_index] += np.array(v)
                index_num[6][vs_lo_vd_index] += 1

                index_sum[7][vd_ld_index] += np.array(v)
                index_num[7][vd_ld_index] += 1

        index_sum /= (index_num + 1e-7)
        log_file = open("test_log/log_vs" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[0]]))
        log_file = open("test_log/log_lo" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[1]]))
        log_file = open("test_log/log_vd" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[2]]))
        log_file = open("test_log/log_il" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[3]]))
        log_file = open("test_log/log_ld" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[4]]))
        log_file = open("test_log/log_vs_lo" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[5]]))
        log_file = open("test_log/log_vs_lo_vd" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[6]]))
        log_file = open("test_log/log_vd_ld" + str(module_n) + ".txt", "wt")
        log_file.write("\n".join([ "\t".join([str(t) for t in s]) for s in index_sum[7]]))     