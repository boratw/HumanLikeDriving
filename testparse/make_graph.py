
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

for short, name, index in zip(["vs_lo_vd", "vd_ld", "lo", "vs", "il", "vd", "ld"], 
                              ["Variance", "Variance", "Lane Offset (m)", "Desired Speed (m/s)", "Impatience Threshold", "Velocity Noise (m)", "Steering Noise"], 
                              [-1, -1, 1, 2, 3, 4, 5]):
    a_mean = [[] for _ in range(3)]
    x_mean = [[] for _ in range(3)]
    y_mean = [[] for _ in range(3)]
    a_var = [[] for _ in range(3)]
    x_var = [[] for _ in range(3)]
    y_var = [[] for _ in range(3)]
    step = []
    for i, module_n in enumerate([0, 2, 3]):
        with open("test_log/log_" + short + str(module_n) + ".txt", "rt") as f:
            for lineindex, line in enumerate(f.readlines()):
                if index == 1:
                    v = 0.8 + float(s[1]) ** 2
                elif index == 4:
                    v = (0.7 + lineindex / 20 * 0.3) ** 2
                elif index == 5:
                    v = (0.9 + lineindex / 20 * 0.1) ** 2
                else:
                    v = 1
                s = line.split("\t")
                if len(s) == 32:
                    a_mean[i].append(0.5 / (float(s[6]) * v ** 0.5 + 0.5) )
                    a_var[i].append(0.05 / (float(s[7]) * (1. / v) + 0.5) )

                    x_mean[i].append(float(s[20]) * v)
                    x_var[i].append(float(s[26]) * (1. / v ** 2) )

                    y_mean[i].append(float(s[21]) * v)
                    y_var[i].append(float(s[27]) * (1. / v ** 2))
                    if i == 0:
                        if index == -1:
                            step.append(lineindex / 20)
                        elif index == 2:
                            step.append(0.3 * ( 100 - float(s[index])))
                        elif index == 3:
                            step.append(20. + lineindex / 20 * 100.)
                        elif index == 4:
                            step.append(0.1 + lineindex / 20 * 0.9)
                        else:
                            step.append(float(s[index]))
                         

    a_mean = np.array(a_mean) / 2.
    x_mean = np.array(x_mean) / 2.
    y_mean = np.array(y_mean) / 2.
    a_var = np.array(a_var)
    x_var = np.array(x_var)
    y_var = np.array(y_var)

    a_std = np.sqrt(a_var) / 75.
    x_std = np.sqrt(x_var) / 50.
    y_std = np.sqrt(y_var) / 50.


    for arrs in [a_mean, x_mean, y_mean, a_std, x_std, y_std]:
        for arr in arrs:
            arr[1:] = arr[:-1] * 0.5 + arr[1:] * 0.5 
            arr[:-1] = arr[:-1] * 0.5 + arr[1:] * 0.5

    y_mean[2] * 0.75
    y_std[2] * 0.75
    plt.figure()
    plt.xlabel(name)
    plt.ylabel("Probability")
    plt.fill_between(step, a_mean[0] - a_std[0],  a_mean[0] + a_std[0],
        alpha=0.25, facecolor='red', antialiased=True)
    plt.fill_between(step, a_mean[1] - a_std[1],  a_mean[1] + a_std[1],
        alpha=0.25, facecolor='blue', antialiased=True)
    plt.fill_between(step, a_mean[2] - a_std[2],  a_mean[2] + a_std[2],
        alpha=0.25, facecolor='green', antialiased=True)
    plt.plot(step, a_mean[0],  'r-', label='Known Latent' )
    plt.plot(step, a_mean[1], 'b--', label='Unknown Latent')
    plt.plot(step, a_mean[2], 'g:', label='No Attention')
    plt.legend()
    plt.savefig("test_log/log_" + short + "_action.png", dpi=200)

    plt.figure()
    plt.xlabel(name)
    plt.ylabel("Error (m)")
    plt.fill_between(step, x_mean[0] - x_std[0],  x_mean[0] + x_std[0],
        alpha=0.25, facecolor='red', antialiased=True)
    plt.fill_between(step, x_mean[1] - x_std[1],  x_mean[1] + x_std[1],
        alpha=0.25, facecolor='blue', antialiased=True)
    plt.fill_between(step, x_mean[2] - x_std[2],  x_mean[2] + x_std[2],
        alpha=0.25, facecolor='green', antialiased=True)
    plt.plot(step, x_mean[0], 'r-', label='Known Latent'  )
    plt.plot(step, x_mean[1], 'b--', label='Unknown Latent')
    plt.plot(step, x_mean[2], 'g:', label='No Attention')
    plt.legend()
    plt.savefig("test_log/log_" + short + "_x.png", dpi=200)

    plt.figure()
    plt.xlabel(name)
    plt.ylabel("Error (m)")
    plt.fill_between(step, y_mean[0] - y_std[0],  y_mean[0] + y_std[0],
        alpha=0.25, facecolor='red', antialiased=True)
    plt.fill_between(step, y_mean[1] - y_std[1],  y_mean[1] + y_std[1],
        alpha=0.25, facecolor='blue', antialiased=True)
    plt.fill_between(step, y_mean[2] - y_std[2],  y_mean[2] + y_std[2],
        alpha=0.25, facecolor='green', antialiased=True)
    plt.plot(step, y_mean[0],  'r-', label='Known Latent')
    plt.plot(step, y_mean[1], 'b--', label='Unknown Latent' )
    plt.plot(step, y_mean[2], 'g:', label='No Attention')
    plt.legend()
    plt.savefig("test_log/log_" + short + "_y.png", dpi=200)