import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 225
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

data_name_list = [r"DHF",r"No-DHF",r"CHG",r"IFG"]

# Create subplots
fig, axs = plt.subplots(1, 2)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
folder_s = os.path.join(get_controller_path(), "sim_alg/test_online/log-test_online_static/test_online_static-2024-December-14-12-10-32-ail")
folder_g = os.path.join(get_controller_path(), "sim_alg/test_online/log-test_cmp_ft_ae_hg/test_cmp_ft_ae_hg-2024-December-14-13-15-06-ail")

data_list = []

for i in range(11):
    ne_file = os.path.join(folder_s,"ES_GGM.nc.final.txt")
    ne = np.genfromtxt(ne_file, delimiter=',')[:,3:]
    ne = ne.squeeze()
    data_list.append(ne[i::11])


for i in range(11):
    nf_file = os.path.join(folder_s,"ES_GGM.nf.final.txt")
    nf = np.genfromtxt(nf_file, delimiter=',')[:,3:]
    nf = nf.squeeze()/1000
    data_list.append(nf[i::11])

data_list_g = []
data_list_g.append(data_list[8])


ne_file = os.path.join(folder_g,"ES_GGM.ae.nc.final.txt")
ne = np.genfromtxt(ne_file, delimiter=',')[:,3:]
ne = ne.squeeze()
data_list_g.append(ne)

ne_file = os.path.join(folder_g,"ES_GGM.ch.nc.final.txt")
ne = np.genfromtxt(ne_file, delimiter=',')[:,3:]
ne = ne.squeeze()
data_list_g.append(ne)

ne_file = os.path.join(folder_g,"ES_GGM.if.nc.final.txt")
ne = np.genfromtxt(ne_file, delimiter=',')[:,3:]
ne = ne.squeeze()
data_list_g.append(ne)

data_list_g.append(data_list[8+11])

nf_file = os.path.join(folder_g,"ES_GGM.ae.nf.final.txt")
nf = np.genfromtxt(nf_file, delimiter=',')[:,3:]
nf = nf.squeeze()/1000
data_list_g.append(nf)

nf_file = os.path.join(folder_g,"ES_GGM.ch.nf.final.txt")
nf = np.genfromtxt(nf_file, delimiter=',')[:,3:]
nf = nf.squeeze()/1000
data_list_g.append(nf)

nf_file = os.path.join(folder_g,"ES_GGM.if.nf.final.txt")
nf = np.genfromtxt(nf_file, delimiter=',')[:,3:]
nf = nf.squeeze()/1000
data_list_g.append(nf)



lines = []

xlabels = [r'$Z$',r'$\Sigma_{k}  \mathbf{1}_{\{r_k < \hat{r}\}}/K$']

markers = ['x','o','+','s','d']

for i in range(2):
    for j in range(4):
        data_sorted = np.sort(data_list_g[j+i*4])
        print(data_sorted)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        line, = axs[i].plot(data_sorted,cdf,marker=markers[j],markevery=25,linewidth=1.25,markerfacecolor='none')
        lines.append(line)
        
    axs[i].set_xlabel(xlabels[i])
    axs[i].set_yticks(np.arange(0,6)*0.2)      # Remove y-axis ticks
    axs[i].set_ylim(0, 1)

    if i == 0:
        axs[i].set_ylabel(r'CDF')
    else:
        axs[i].set_yticklabels([])
        axs[i].set_ylabel('')      # Remove y-axis label

    if i == 0:
        axs[i].set_xlim(20, 50)
        axs[i].set_xticks([20,30,40,50])     
    if i == 1:
        axs[i].set_xlim(0, 0.04)
        axs[i].set_xticks([0,0.02,0.04])      
        axs[i].set_xticklabels(['0','0.02','0.04'])

    axs[i].set_position([0.175+0.425*i, 0.195, 0.325, 0.665])
    axs[i].grid(True)

# Add a legend
fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.175, 0.875, 0.75, 0.085), ncol = 4 , borderaxespad=0.1,handlelength=1.2,fancybox=True, framealpha=1,mode='expand' )
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()