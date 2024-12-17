import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 275
dpi = 100  # Typical screen DPI, adjust if necessary
fig_width_in = fig_width_px / dpi
fig_height_in = fig_height_px / dpi

plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
plt.rc('font', size=FONT_SIZE)  # Default font size
plt.rc('axes', titlesize=FONT_SIZE)  # Font size of the axes title
plt.rc('axes', labelsize=FONT_SIZE)  # Font size of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)  # Font size of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)  # Font size of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)  # Font size for legends

def moving_average(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='full')


from working_dir_path import get_controller_path
current_dir = os.path.dirname(os.path.abspath(__file__))

data_name_list = ["FT","No-FT","ALLE"]

# Create subplots
fig, axs = plt.subplots(2, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
folder = [
    os.path.join(get_controller_path(), "sim_alg/test_online/log-test_mobile_hf/test_mobile_hf-2024-December-16-02-15-58-ail"),
    os.path.join(get_controller_path(), "sim_alg/test_online/log-test_mobile_hf_nft/test_mobile_hf_nft-2024-December-16-01-21-12-ail"),
    os.path.join(get_controller_path(), "sim_alg/test_online/log-test_mobile_ae/test_mobile_ae-2024-December-16-00-34-54-ail"),
]

data_list = []

path = []
path.append(os.path.join(folder[0],"ES_GGM.bler.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.bler.final.txt"))
path.append(os.path.join(folder[2],"ES_GGM.bler.final.txt"))
for idx, data_file in enumerate(path):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = np.genfromtxt(data_file, delimiter=',')[:,3:5]
    # Step 1: Extract x and y columns
    x = data[:, 0]
    y = data[:, 1]

    # Step 2: Find unique y values and get inverse indices
    unique_y, inverse_indices = np.unique(y, return_inverse=True)

    # Step 3: Compute the sum of x for each unique y
    sum_x = np.bincount(inverse_indices, weights=x)

    # Step 4: Count the number of occurrences for each unique y
    count = np.bincount(inverse_indices)

    # Step 5: Calculate the mean of x for each unique y
    mean_x = sum_x / count
    data_list.append(np.log10(mean_x))

path = []
path.append(os.path.join(folder[0],"ES_GGM.nc.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.nc.final.txt"))
path.append(os.path.join(folder[2],"ES_GGM.nc.final.txt"))
for idx, data_file in enumerate(path):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = np.genfromtxt(data_file, delimiter=',')[:,3:5]
    # Step 1: Extract x and y columns
    x = data[:, 0]
    y = data[:, 1]

    # Step 2: Find unique y values and get inverse indices
    unique_y, inverse_indices = np.unique(y, return_inverse=True)

    # Step 3: Compute the sum of x for each unique y
    sum_x = np.bincount(inverse_indices, weights=x)

    # Step 4: Count the number of occurrences for each unique y
    count = np.bincount(inverse_indices)

    # Step 5: Calculate the mean of x for each unique y
    mean_x = sum_x / count
    data_list.append(mean_x)

lines = []

ylabels = [r'$\mathbb{E}[r_k]$',r'$\mathbb{E}[Z]$']
markers = ['x','o','s','+','d']

for i in range(2):
    for j in range(3):
        line, = axs[i].plot(data_list[j+i*3],marker=markers[j],linewidth=1,markerfacecolor='none')
        lines.append(line)
    axs[i].set_ylabel(ylabels[i])

    if i == 0:
        axs[i].set_xlabel(r'Mobility of STAs (meter/second)')
        axs[i].set_xticks(np.arange(0,11))      
        axs[i].set_xticklabels(np.arange(0,11)/2)
    else:
        axs[i].set_xticks(np.arange(0,11))      
        axs[i].set_xticklabels([])

    if i == 0:
        axs[i].set_ylim(-3.5, -1.5)
        axs[i].set_yticks([-3,-2])    
        axs[i].set_yticklabels([r'$10^{-3}$',r'$10^{-2}$'])    

    if i == 1:
        axs[i].set_ylim(0, 50)
        axs[i].set_yticks([0,25,50])      

    axs[i].set_position([0.175, 0.15+i*0.4, 0.75, 0.35])
    axs[i].grid(True)

fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.905, 0.75, 0.1), ncol = 3 , borderaxespad=0.1,handlelength=1.5,fancybox=True, framealpha=1,mode='expand' )


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()