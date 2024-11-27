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

data_name_list = ["STNN","PCNN","PHNN"]

log_path_list = []
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_tokenizer/selected_nn/tokenizer_base.loss.final.txt"))
# log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_sparser/selected_nn/sparser_base.loss.final.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_predictor/selected_nn/PCNN.loss.final.txt"))
log_path_list.append(os.path.join(get_controller_path(),"sim_alg/train_predictor/selected_nn/PHNN.loss.final.txt"))
# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

markers = ['v', '^', 'o','+']  # Different markers for each line
# p_names = [r'a',r'b']
lines = []

for t in range(len(log_path_list)):
    data_file = os.path.join(log_path_list[t])
    data = np.genfromtxt(data_file, delimiter=',')
    line, = axs.plot(data[:,3]/data[0,3],linewidth=1,markerfacecolor='none')
    lines.append(line)

axs.set_position([0.175, 0.265, 0.775, 0.675])
# Add labels and title
axs.set_xlabel(r'Iterations')
axs.grid(True)



# axs[0].set_ylabel(r'Number of iterations')
axs.set_ylabel(r'Normalized Loss')
# # uu = 5*20
# # ll = 15*20
# axs[0].set_xlim(5*20, 10*20)
# axs[1].set_xlim(5*20, 10*20)
#
axs.set_ylim(0, 1.05)
# # axs[2].set_xlim(uu, ll)
# axs.set_ylim(0, 200)
# axs.set_xlim(50, 700)
# axs[1].set_ylim(0, 10)
# axs[0].set_yticks([0,2,4,6,8,10])
# axs.set_xticks(100*np.arange(8))
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')


# Add a legend
fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.65, 0.5, 0.2, 0.1),ncol = 1 ,borderaxespad=0.1,handlelength=1.5,fancybox=True, framealpha=1)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()