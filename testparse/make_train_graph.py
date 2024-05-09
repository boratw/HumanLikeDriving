
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

arr = []
with open("train_log/DrivingStyle_AvgLatent4/2024-03-13-14-05-16.txt", "rt") as f:
    for lineindex, line in enumerate(f.readlines()[1:100]):
        s = line.split("\t")
        if len(s) > 100:
            arr.append([float(t) for t in s if t != '\n'])
                    

    arr = np.array(arr)
    steps = arr[:, 0]
    steps *= 64

    r1 = arr[:, 1] +  arr[:, 2]
    a1 = arr[:, 7]
    r2 = arr[:, 8] + arr[:, 9]
    a2 = arr[:, 14]
    r3 = arr[:, 15] + arr[:, 16]
    a3 = arr[:, 21]

    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Route Loss")
    plt.plot(steps, r2,  'r-', label='With Attention' )
    plt.plot(steps, r1, 'b--', label='Without Attention')
    plt.legend()
    plt.savefig("test_log/train_log_route.png", dpi=200)

    plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Action Loss")
    plt.plot(steps, a1,  'r-', label='With Attention' )
    plt.plot(steps, a2, 'b--', label='Without Attention')
    plt.legend()
    plt.savefig("test_log/train_log_action.png", dpi=200)
