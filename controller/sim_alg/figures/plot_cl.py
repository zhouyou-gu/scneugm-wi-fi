import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 350
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

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='full')


from working_dir_path import get_controller_path
current_dir = os.path.dirname(os.path.abspath(__file__))

data_name_list = ["Adp-B","Lin-B","No-B"]

# Create subplots
fig, axs = plt.subplots(3, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
folder = [
    os.path.join(get_controller_path(), "sim_alg/train_curriculum_learning/log-train_es_ggm_cl_rand_init/train_es_ggm_cl_rand_init-2024-December-12-20-47-00-ail"),
    os.path.join(get_controller_path(), "sim_alg/train_curriculum_learning/log-train_es_ggm_cl_rand_init_nadaptive/train_es_ggm_cl_rand_init_nadaptive-2024-December-13-13-29-52-ail"),
    os.path.join(get_controller_path(), "sim_alg/train_curriculum_learning/log-train_es_ggm_cl_rand_init_all/train_es_ggm_cl_rand_init_all-2024-December-12-21-54-51-ail"),
]

data_list = []

for idx, f in enumerate(folder):
    nc_file = os.path.join(f,"ES_GGM.nc.final.txt")
    nc = np.genfromtxt(nc_file, delimiter=',')[:,3:]
    nc = nc.squeeze()

    ub_nc_file = os.path.join(f,"ES_GGM.ub_nc.final.txt")
    ub_nc = np.genfromtxt(ub_nc_file, delimiter=',')[:,3:]
    ub_nc = ub_nc.squeeze()
    
    q_avg_file = os.path.join(f,"ES_GGM.q_avg.final.txt")
    q_avg = np.genfromtxt(q_avg_file, delimiter=',')[:,3:]
    q_avg = q_avg.squeeze()

    rwd = (q_avg>0.99)*(nc<=ub_nc)
    rwd = rwd.astype(float)
    rwd = moving_average(rwd)
    data_list.append(rwd)

    

path = []
path.append(os.path.join(folder[0],"ES_GGM.reward.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.reward.final.txt"))
path.append(os.path.join(folder[2],"ES_GGM.reward.final.txt"))
for idx, data_file in enumerate(path):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = np.genfromtxt(data_file, delimiter=',')[:,2]
    data = data-data[0]
    data = data.squeeze()/60
    data_list.append(data)
    print(data)
    
path = []
path.append(os.path.join(folder[0],"ES_GGM.n_sta.final.txt"))
path.append(os.path.join(folder[1],"ES_GGM.n_sta.final.txt"))
path.append(os.path.join(folder[2],"ES_GGM.n_sta.final.txt"))
for idx, data_file in enumerate(path):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    data = np.genfromtxt(data_file, delimiter=',')[:,3:]
    data = data.mean(axis=1).squeeze()
    data_list.append(data)

lines = []

ylabels = [r'$\dot{R}\geq 0$',r'Time (mins)',r"$K'$",]
for i in range(3):
    for j in range(3):
        line, = axs[i].plot(data_list[j+i*3],linewidth=1.5,markerfacecolor='none')
        lines.append(line)
    axs[i].set_ylabel(ylabels[i])

    if i == 0:
        axs[i].set_xlabel(r'Iterations')
        # axs[i].set_yticks(np.arange(4, num_rows, tick_step))
        # axs[i].set_yticklabels(np.arange(4, num_rows, tick_step)+1)
    else:
        axs[i].set_xticks(np.arange(0,1001,200))      # Remove y-axis ticks
        axs[i].set_xticklabels([])
        axs[i].set_xlabel('')      # Remove y-axis label

    if i == 0:
        axs[i].set_ylim(-0.1, 1)

    if i == 1:
        axs[i].set_ylim(-5, 60)
        axs[i].set_yticks([0,30,60])      # Remove y-axis ticks

    if i == 2:
        axs[i].set_ylim(-50, 1100)
        axs[i].set_yticks([0,500,1000])      # Remove y-axis ticks

    axs[i].set_xlim(0, 1092)
    axs[i].set_position([0.175, 0.11+i*0.275, 0.75, 0.25])
    axs[i].grid(True)

# Add a legend
fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.915, 0.75, 0.1), ncol = 3 , borderaxespad=0.1,handlelength=1.5,fancybox=True, framealpha=1,mode='expand' )
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()