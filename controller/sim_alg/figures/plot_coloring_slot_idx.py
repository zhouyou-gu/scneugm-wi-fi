import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 200
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

data_name_list = ["STAs","QoS Violations"]

# Create subplots
fig, axs = plt.subplots(1, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
folder = [
    os.path.join(get_controller_path(), "sim_alg/test_online/log-test_online_static_coloring_idx_test/test_online_static_coloring_idx_test-2024-December-17-14-41-47-ail"),
]

data_list = []

path = []
path.append(os.path.join(folder[0],"ES_GGM.col_idx.final.txt"))
path.append(os.path.join(folder[0],"ES_GGM.fai_idx.final.txt"))
for idx, data_file in enumerate(path):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = np.genfromtxt(data_file, delimiter=',')[:,3:]
    data_list.append(np.mean(data[0::11,:35],axis=0))
print(data_list[1])
lines = []

ylabels = [r'$\mathbb{E}[r_k]$',r'$\mathbb{E}[Z]$']
markers = ['x','o','s','+','d']
indices = np.arange(35)
width = 0.35  # Bar width

b = axs.bar(indices-width/2, data_list[0], width, color = "deepskyblue", label='Array 1')
lines.append(b)
axs.set_ylabel('Number')
axs.set_ylim(0,60)
axs.set_yticks([0,15,30,45,60])
axs.set_xlim(-2,32)
axs.tick_params(axis='y', colors='deepskyblue')
axs.yaxis.label.set_color('deepskyblue')
axs.spines['left'].set_color('deepskyblue')

raxs = axs.twinx()
b = raxs.bar(indices + width/2, data_list[1], width, color= "tomato", label='Array 2')
raxs.set_ylabel('Number')
lines.append(b)
raxs.tick_params(axis='y', colors='tomato')
raxs.yaxis.label.set_color('tomato')
raxs.spines['right'].set_color('tomato')

axs.set_position([0.175, 0.2, 0.625, 0.625])
axs.grid(True)
axs.set_xlabel("Slot indices")

axs.spines['right'].set_visible(False)
raxs.spines['left'].set_visible(False)

fig.legend([lines[0],lines[1]], data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.85, 0.625, 0.1), ncol = 3 , borderaxespad=0.1,handlelength=1.5,fancybox=True, framealpha=1,mode='expand' )

# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()