import numpy as np
import matplotlib.pyplot as plt
import os

prob_mean = np.zeros((3, 40))
prob_var = np.zeros((3, 40))
x_mean = np.zeros((3, 40))
x_var = np.zeros((3, 40))
y_mean = np.zeros((3, 40))
y_var = np.zeros((3, 40))

add_num = 0

exp_names = os.listdir("../test_log/log4_5")
for exp_name in exp_names:
    if exp_name.startswith("module1") and \
        exp_name.find("VR8") != -1:
        with open('../test_log/log4_5/' + exp_name) as f:
            latent1 = []
            latent2 = []
            latent3 = []
            latent4 = []
            dx = []
            dy = []
            for line in f.readlines()[:400]:
                sep = line.split('\t')
                latent1.append(float(sep[0]))
                latent2.append(float(sep[1]))
                latent3.append(float(sep[2]))
                latent4.append(float(sep[3]))
                dx.append(float(sep[5]))
                dy.append(float(sep[6]))

            plt.figure()
            # X locations for the groups
            steps = np.arange(0, 2000, 5)

            # Create figure and axis
            fig, ax = plt.subplots()

            plt.plot(steps, latent1, label="l0", color="red")
            plt.plot(steps, latent2, label="l1", color="green")
            plt.plot(steps, latent3, label="l2", color="blue")
            plt.plot(steps, latent4, label="l3", color="magenta")


            plt.xlabel("Step (Time)")
            plt.ylabel("Latents")
            plt.title("Latent Value Over Time")
            plt.legend()

            # Display the chart
            plt.tight_layout()
            plt.savefig(exp_name + "_latent.png", dpi=300, bbox_inches="tight")

            plt.figure()
            # Create figure and axis
            fig, ax = plt.subplots()

            plt.plot(steps, dx, label="Lateral Error (m)", color="red")
            plt.plot(steps, dy, label="Longitudinal Error (m)", color="blue")

            plt.xlabel("Step (Time)")
            plt.ylabel("Error")
            plt.title("Error Over Time")
            plt.legend()

            # Display the chart
            plt.tight_layout()
            plt.savefig(exp_name + "_error.png", dpi=300, bbox_inches="tight")
