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

data_name_list = ["GGM-A","GGM-B"]

# Create subplots
fig, axs = plt.subplots(1, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
folder = [
    os.path.join(get_controller_path(), "sim_alg/test_online/log-test_mobile_ae_profiling/test_mobile_ae_profiling-2024-December-17-13-43-15-ail"),
    os.path.join(get_controller_path(), "sim_alg/test_online/log-test_mobile_hf_profiling/test_mobile_hf_profiling-2024-December-17-13-43-36-ail"),
]

data_list = []

path = []
path.append(os.path.join(folder[0],"ES_GGM.tim_tok.final.txt"))
path.append(os.path.join(folder[0],"ES_GGM.tim_dhf.final.txt"))
path.append(os.path.join(folder[0],"ES_GGM.tim_tab.final.txt"))
path.append(os.path.join(folder[0],"ES_GGM.tim_pdt.final.txt"))
path.append(os.path.join(folder[0],"ES_GGM.tim_ggm.final.txt"))
path.append(os.path.join(folder[0],"ES_GGM.tim_col.final.txt"))
for idx, data_file in enumerate(path):
    if not os.path.exists(data_file):
        data_list.append(0.)
        continue
    data = np.genfromtxt(data_file, delimiter=',')[3]
    data = data.squeeze()
    data_list.append(data/1000)

path = []
path.append(os.path.join(folder[1],"ES_GGM.tim_tok.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.tim_dhf.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.tim_tab.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.tim_pdt.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.tim_ggm.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.tim_col.final.txt"))
for idx, data_file in enumerate(path):
    if not os.path.exists(data_file):
        data_list.append(0.)
        continue
    data = np.genfromtxt(data_file, delimiter=',')[3]
    data = data.squeeze()
    data_list.append(data/1000)

lines = []

ylabels = [r'$\mathbb{E}[r_k]$',r'$\mathbb{E}[Z]$']
markers = ['x','o','+','s','d']

colors = ['lightgray', 'darkred', 'lightpink', 'darkgreen', 'lightcyan', 'darkorange']
hatches = ['\\\\', '//', '\\\\', '//', '\\\\', '//']

start_times = [0,0]
for i in range(6):
    b = axs.barh(data_name_list, [data_list[i],data_list[i+6]], left=start_times, height=0.5, color=colors[i], edgecolor='black', hatch=hatches[i])
    start_times[0] += data_list[i]
    start_times[1] += data_list[i+6]
    lines.append(b)

axs.set_position([0.175, 0.2, 0.75, 0.55])
axs.grid(True,axis='x')
axs.set_xlabel("Processing time (millisecond)")
box_names = ["Tok.", "Hsh.", "Buc.", "Pre.", "EG", "Col."]
fig.legend(lines, box_names ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.775, 0.75, 0.1), ncol = 3 , borderaxespad=0.1,handlelength=1.75,fancybox=True, framealpha=1,mode='expand' )



# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()