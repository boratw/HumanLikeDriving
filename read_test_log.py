
import numpy as np
import cv2

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

v_at_step = [0. for _ in range(2001)]
v_at_step_num = [0 for _ in range(2001)]

map_image = cv2.imread("visualizer/lanemap_batjeon.png")


log_file = open("policy_test_log/result_0706/zreaded_sff_policy_drivestyle_256.txt", "wt")
log_file_v = open("policy_test_log/result_0706/zreaded_sff_policy_drivestyle_256.txt", "wt")
for index in range(404):
    step = 0
    v_sum = 0
    readed_sum = 0
    screen = np.zeros((512, 512, 3), np.uint8)
    agentpos = []
    npcpos = []
    with open("policy_test_log/result_0706/sff_policy_drivestyle_256_" + str(index) + ".txt", "rt") as f:
        lines = f.readlines()
        length = len(lines)
        print(length)
        prevx = 0.
        prevy = 0.
        yaw  = 0.
        yawcos = 1.
        yawsin = 0.
        for i, line in enumerate(lines):
            split = line.split("\t")
            if len(split) >= 17:
                v_sum += float(split[2])
                v_at_step[step] += float(split[2])
                v_at_step_num[step] += 1
                step += 1
            if len(split) >= 18:
                readed_sum += float(split[17])
                x = float(split[18])
                y = float(split[19])
                if ((prevx - x) ** 2 + (prevy - y) ** 2) > 0.01:
                    yaw = np.arctan2(y - prevy, x - prevx)
                    yawcos = np.cos(-yaw)
                    yawsin = np.sin(-yaw)
                if length < 1990:
                    if (length - i) < 100 and (length - i) % 5 == 1:
                        agentpos.append([x, y])
                        npc = []
                        for j in range(20, len(split), 2):
                            px, py = rotate(float(split[j]), float(split[j+1]), yawsin, yawcos)
                            npc.append([px, py])
                        npcpos.append(npc)
                prevy = y
                prevx = x

        if length < 1990:
            M = cv2.getRotationMatrix2D((512, 512), 90, 1.0)

            for i in range(len(agentpos)):
                c = int(12.7 * (21 + i - len(agentpos)))
                cv2.circle(screen, (int((agentpos[i][0] - agentpos[-1][0]) * 8) + 256, int((agentpos[i][1] - agentpos[-1][1]) * 8) + 256), 4, (0, c, 0), -1)
                for npc in npcpos[i]:
                    cv2.circle(screen, (int((npc[0] + agentpos[i][0] - agentpos[-1][0]) * 8) + 256, int((npc[1] + agentpos[i][1] - agentpos[-1][1]) * 8) + 256), 4, (0, 0, c), -1)

            cv2.imwrite("policy_test_log/result_0706/sff_policy_drivestyle_256_" + str(index) + ".png", screen)


            
    if step > 100:
        log_file.write(str(step) + "\t" + str(v_sum / step) + "\t" + str(readed_sum / step) + "\n")

for i in range(2001):
    if v_at_step_num[i] > 0:
        log_file_v.write(str(v_at_step[i] / v_at_step_num[i]) + "\n")