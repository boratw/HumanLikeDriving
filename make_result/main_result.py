import numpy as np
import matplotlib.pyplot as plt

prob_mean = np.zeros((3, 3))
prob_var = np.zeros((3, 3))
x_mean = np.zeros((3, 3))
x_var = np.zeros((3, 3))
y_mean = np.zeros((3, 3))
y_var = np.zeros((3, 3))

for module in range(3):
    with open('../test_log/log4_5/module' + str(module) + '.txt') as f:
        for i, line in enumerate(f.readlines()):
            sep = line.split('\t')
            ran = i % 4 - 1
            if ran != -1:
                prob_mean[module][ran] += float(sep[1]) / 23
                prob_var[module][ran] += float(sep[2]) / 10. / 23
                x_mean[module][ran] += float(sep[3]) / 23
                x_var[module][ran] += float(sep[9]) / 10. / 23
                y_mean[module][ran] += float(sep[4]) / 23
                y_var[module][ran] += float(sep[10]) / 10. / 23

print(prob_mean)
print(prob_var)
print(x_mean)
print(x_var)
print(y_mean)
print(y_var)


# Data for the three models
models = ['No Latent', 'No Prior', 'With Prior']
cases = ['Low Randomness', 'Medium Randomness', 'High Randomness']

'''
# Mean scores for each model under the three cases
mean_scores = {
    'No Latent': [0.8, 0.6, 0.7],
    'No Prior': [0.75, 0.65, 0.6],
    'With Prior': [0.85, 0.7, 0.75]
}

# Standard deviations for each model under the three cases
stddev_scores = {
    'No Latent': [0.05, 0.07, 0.06],
    'No Prior': [0.04, 0.06, 0.05],
    'With Prior': [0.03, 0.05, 0.04]
}
'''
# Bar width
bar_width = 0.2

for name, m, d in zip(['Action Probability', 'Lateral Error (m)', 'Longitudinal Error (m)'], [prob_mean, x_mean, y_mean], [prob_var, x_var, y_var]):
    plt.figure()
    # X locations for the groups
    x = np.arange(len(cases))

    # Create figure and axis
    fig, ax = plt.subplots()

    # Plotting the bars with error bars representing standard deviation
    ax.bar(x - bar_width, m[0], bar_width, yerr=d[0], label='No Latents', capsize=5)
    ax.bar(x, m[1], bar_width, yerr=d[1], label='Unknown Latents', capsize=5)
    ax.bar(x + bar_width, m[2], bar_width, yerr=d[2], label='Known Latents', capsize=5)

    # Adding labels and title
    ax.set_xlabel('Randomness Cases')
    ax.set_ylabel(name)
    ax.set_title('Comparison of Models with Different Randomness Cases')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()

    # Display the chart
    plt.tight_layout()
    plt.savefig("main " + name + ".png", dpi=300, bbox_inches="tight")