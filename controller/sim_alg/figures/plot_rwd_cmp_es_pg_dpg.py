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




data_name_list = ["ES","PG","DPG"]

log_path_list = []
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_vs_pg_dpg/log-train_es/train_es-2024-November-30-16-43-00-ail/ES_GGM.reward.final.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_vs_pg_dpg/log-train_pg/train_pg-2024-November-30-17-03-06-ail/PG_GGM.reward.final.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_and_test_es_vs_pg_dpg/log-train_dpg/train_dpg-2024-November-30-17-11-33-ail/PG_GGM.reward.final.txt"))

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

markers = ['v', '^', 'o','+']  # Different markers for each line
# p_names = [r'a',r'b']
lines = []

for t in range(len(log_path_list)):
    data_file = os.path.join(log_path_list[t])
    data = np.genfromtxt(data_file, delimiter=',')
    line, = axs.plot(moving_average(data[:,3]>0)[0:1000],linewidth=1.5,markerfacecolor='none')
    lines.append(line)
    # lines.append(line)

axs.set_position([0.175, 0.265, 0.775, 0.675])
# Add labels and title
axs.set_xlabel(r'Iterations')
axs.grid(True)



# axs[0].set_ylabel(r'Number of iterations')
axs.set_ylabel(r'$\dot{R}\geq 0$')

axs.set_ylim(-0.1, 1.0)
axs.set_xlim(0, 1050)


# Add a legend
fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.2, 0.525, 0.2, 0.1),ncol = 1 ,borderaxespad=0.1,handlelength=1,fancybox=True, framealpha=1, columnspacing=0.5)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()