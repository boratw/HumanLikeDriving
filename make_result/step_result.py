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

for module_n in range(3):
    exp_names = os.listdir("../test_log/log4_5")
    for exp_name in exp_names:
        if exp_name.startswith("module" + str(module_n)) and \
            exp_name.endswith("_step.txt") and \
            exp_name.find("VR8") != -1:
            print(exp_name)
            with open('../test_log/log4_5/' + exp_name) as f:
                for i, line in enumerate(f.readlines()[20:340]):
                    ran = i // 8
                    sep = line.split('\t')
                    prob_mean[module_n][ran] += float(sep[0])
                    prob_var[module_n][ran] += float(sep[1]) / 10.
                    x_mean[module_n][ran] += float(sep[2])
                    x_var[module_n][ran] += float(sep[8]) / 10.
                    y_mean[module_n][ran] += float(sep[3])
                    y_var[module_n][ran] += float(sep[9]) / 10.
            if module_n == 0:
                add_num += 8

prob_mean /= add_num
prob_var /= add_num
x_mean /= add_num
x_var /= add_num
y_mean /= add_num
y_var /= add_num

prob_mean[:, 1:-1] = (prob_mean[:, :-2] +  prob_mean[:, 1:-1] +  prob_mean[:, 2:]) / 3
prob_var[:, 1:-1] = (prob_var[:, :-2] +  prob_var[:, 1:-1] +  prob_var[:, 2:]) / 3
x_mean[:, 1:-1] = (x_mean[:, :-2] +  x_mean[:, 1:-1] +  x_mean[:, 2:]) / 3
x_var[:, 1:-1] = (x_var[:, :-2] +  x_var[:, 1:-1] +  x_var[:, 2:]) / 3
y_mean[:, 1:-1] = (y_mean[:, :-2] +  y_mean[:, 1:-1] +  y_mean[:, 2:]) / 3
y_var[:, 1:-1] = (y_var[:, :-2] +  y_var[:, 1:-1] +  y_var[:, 2:]) / 3

print(prob_mean)
print(prob_var)
print(x_mean)
print(x_var)
print(y_mean)
print(y_var)


# Data for the three models
models = ['No Latent', 'No Prior', 'With Prior']
cases = ['Low Randomness', 'Medium Randomness', 'High Randomness']


for name, m, d in zip(['Action Probability', 'Lateral Error (m)', 'Longitudinal Error (m)'], [prob_mean, x_mean, y_mean], [prob_var, x_var, y_var]):
    plt.figure()
    # X locations for the groups
    steps = np.arange(0, 2000, 50)

    # Create figure and axis
    fig, ax = plt.subplots()

    plt.plot(steps, m[0], label="No Latents", color="blue")
    plt.fill_between(steps, m[0] - d[0], m[0] + d[0], color="blue", alpha=0.2)

    plt.plot(steps, m[1], label="Unknown Latents", color="orange")
    plt.fill_between(steps, m[1] - d[1], m[1] + d[1], color="orange", alpha=0.2)

    plt.plot(steps, m[2], label="Known Latents", color="green")
    plt.fill_between(steps, m[2] - d[2], m[2] + d[2], color="green", alpha=0.2)


    plt.xlabel("Step (Time)")
    plt.ylabel(name)
    plt.title("Model Comparison Over Time")
    plt.legend()

    # Display the chart
    plt.tight_layout()
    plt.savefig("step " + name + ".png", dpi=300, bbox_inches="tight")