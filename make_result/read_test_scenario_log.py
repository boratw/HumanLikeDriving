
import numpy as np
import cv2

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

v_at_step = [0. for _ in range(4001)]
v_at_step_num = [0 for _ in range(4001)]

log_file = open("policy_test_log/result_0801/.txt", "wt")
log_file_v = open("policy_test_log/result_0725/zreaded_sff_policy_Default_0_v.txt", "wt")
for index in range(250):
    step = 0
    v_sum = 0
    readed_sum = 0
    screen = np.zeros((512, 512, 3), np.uint8)
    agentpos = []
    npcpos = []
    with open("policy_test_log/result_0725/sff_policy_Default_0_" + str(index) + ".txt", "rt") as f:
        with open("policy_test_log/result_0725/sff_policy_Default_0_" + str(index) + "_v.txt", "wt") as fw:
            lines = f.readlines()
            length = len(lines)
            print(length)
            for i, line in enumerate(lines):
                split = line.split("\t")
                if len(split) >= 17:
                    v = float(split[2])
                    v_sum += v
                    v_at_step[step] += v
                    v_at_step_num[step] += 1
                    step += 1
                if len(split) >= 18:
                    readed_sum += float(split[17])
                    x = float(split[18])
                    y = float(split[19])
                    if length < 3990:
                        if (length - i) < 100 and (length - i) % 5 == 1:
                            agentpos.append([x, y])
                            npc = []
                            for j in range(21, len(split), 3):
                                npc.append([float(split[j]), float(split[j+1])])
                            npcpos.append(npc)
                    prevy = y
                    prevx = x
                    fw.write(str(v))
                    for j in range(21, len(split), 3):
                        fw.write("\t" + str(np.sqrt((float(split[j]) - x) ** 2 + (float(split[j+1]) - y) ** 2)))
                    fw.write("\n")


            
            if length < 3990:
                M1 = np.float32([[1., 0, -1350 + 256 - agentpos[-1][0] * 5.44],[0, 1., -1635 + 256 - agentpos[-1][1] * 5.44]])
                #M = cv2.getRotationMatrix2D((512, 512), 90, 1.0)
                screen = cv2.warpAffine(map_image, M1, (512,512))
                
                for i in range(len(agentpos)):
                    c = int(12.7 * (21 + i - len(agentpos)))
                    cv2.circle(screen, (int((agentpos[i][0] - agentpos[-1][0]) * 5.44) + 256, int((agentpos[i][1] - agentpos[-1][1]) * 5.44) + 256), 4, (0, c, 0), -1)
                    for npc in npcpos[i]:
                        cv2.circle(screen, (int((npc[0] - agentpos[-1][0]) * 5.44) + 256, int((npc[1] - agentpos[-1][1]) * 5.44) + 256), 4, (0, 0, c), -1)

                cv2.imwrite("policy_test_log/result_0725/sff_policy_Default_0_" + str(index) + ".png", screen)


            
    if step > 100:
        log_file.write(str(step) + "\t" + str(v_sum / step) + "\t" + str(readed_sum / step) + "\n")

for i in range(4001):
    if v_at_step_num[i] > 0:
        log_file_v.write(str(v_at_step[i] / v_at_step_num[i]) + "\n")