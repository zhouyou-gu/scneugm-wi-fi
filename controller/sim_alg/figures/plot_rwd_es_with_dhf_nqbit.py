import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 300
fig_height_px = 150
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

from working_dir_path import get_controller_path
current_dir = os.path.dirname(os.path.abspath(__file__))

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='full')




data_name_list = ["Proposed","None-B","Rand-B"]

log_path_list = []
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.1.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.2.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.3.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.4.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.5.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.6.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.7.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.8.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.9.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_with_dhf/log-train_es_ggm_with_dhf_n_qbit/train_es_ggm_with_dhf_n_qbit-2024-November-29-12-22-43-ail/ES_GGM.reward.10.txt"))

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

# Initialize a list to hold your data arrays
data_arrays = []

# Load the data
for data_file in log_path_list:
    data = np.genfromtxt(data_file, delimiter=',')
    # If necessary, reshape your data to 2D
    # For example, if data is flat and represents a 10x10 grid:
    # data = data.reshape((10, 10))
    data_arrays.append(moving_average(data[:,3]>0)[0:1000].squeeze())

data_arrays = np.asarray(data_arrays)

im = axs.imshow(data_arrays, cmap='plasma', aspect='auto', vmin=0,vmax=0.6)

num_rows = 10
num_cols = 1000

axs.set_position([0.15, 0.265, 0.7, 0.675])
# Add labels and title
axs.set_xlabel(r'Iterations')
# axs.grid(True)
axs.set_yticks(np.arange(1, 10, 2))
axs.set_yticklabels(np.arange(1, 10, 2)+1)

axs.set_xticks(np.arange(0, 6)*200-0.5)
axs.set_xticklabels(np.arange(0, 6)*200)

axs.set_xlim(-0.5, num_cols - 0.5)
axs.set_ylim(num_rows - 0.5, -0.5)  # Invert y-axis if you want row 0 at the top

# axs[0].set_ylabel(r'Number of iterations')
axs.set_ylabel(r'$\Upsilon$')


cbar_ax = fig.add_axes([0.875, 0.265, 0.02, 0.675])  # Position for the colorbar
cbar =fig.colorbar(im, cax=cbar_ax)
cbar.set_ticks([0, 0.3, 0.6])

# Add a legend
# fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.2, 0.525, 0.2, 0.1),ncol = 1 ,borderaxespad=0.1,handlelength=1,fancybox=True, framealpha=1, columnspacing=0.5)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()