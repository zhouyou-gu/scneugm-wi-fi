import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 175
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

data_name_list = ["Edge Count","Recall","Efficiency"]

# Create subplots
fig, axs = plt.subplots(1, 3)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your data files
data_files = [
    os.path.join(get_controller_path(), "sim_alg/test_sparser/log-test_sparser/test_sparser-2024-November-28-09-51-13-ail/edge_proportion.txt"),
    # os.path.join(get_controller_path(), "sim_alg/test_sparser/log-test_sparser/test_sparser-2024-November-28-09-51-13-ail/precision.txt"),
    os.path.join(get_controller_path(), "sim_alg/test_sparser/log-test_sparser/test_sparser-2024-November-28-09-51-13-ail/recall.txt")
]

# Initialize a list to hold your data arrays
data_arrays = []

# Load the data
for data_file in data_files:
    data = np.genfromtxt(data_file, delimiter=',')
    # If necessary, reshape your data to 2D
    # For example, if data is flat and represents a 10x10 grid:
    # data = data.reshape((10, 10))
    data_arrays.append(data.T)

data_arrays.append(data_arrays[1]-data_arrays[0])

# Compute global vmin and vmax for all images
vmin = min(data_array.min() for data_array in data_arrays)
vmax = max(data_array.max() for data_array in data_arrays)

# Plot the 2D grid images
for i in range(3):
    im = axs[i].imshow(data_arrays[i], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    axs[i].set_title(data_name_list[i])
    axs[i].set_xlabel(r'$\Upsilon$')



    # Get the number of rows and columns
    num_rows, num_cols = data_arrays[i].shape

    # Define the interval for ticks
    tick_step = 5  # Set tick interval to every 5 units (adjust as needed)
    
    # Set x and y labels
    if i == 0:
        axs[i].set_ylabel(r'$\Psi$')
        axs[i].set_yticks(np.arange(4, num_rows, tick_step))
        axs[i].set_yticklabels(np.arange(4, num_rows, tick_step)+1)
    else:
        axs[i].set_yticks([])      # Remove y-axis ticks
        axs[i].set_ylabel('')      # Remove y-axis label
    # Replace the existing tick settings with the following
    axs[i].set_xticks(np.arange(4, num_cols, tick_step))

    # Update tick labels accordingly
    axs[i].set_xticklabels(np.arange(4, num_cols, tick_step)+1)

    # Rotate x-axis tick labels if needed
    # plt.setp(axs[i].get_xticklabels(), rotation=45, ha='right')

    # Adjust axis limits
    axs[i].set_xlim(-0.5, num_cols - 0.5)
    axs[i].set_ylim(num_rows - 0.5, -0.5)  # Invert y-axis if you want row 0 at the top

    # # Add a colorbar to each subplot
    # cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
    # cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")
    
    axs[i].set_position([0.125+i*0.25, 0.225, 0.225, 0.625])

cbar_ax = fig.add_axes([0.875, 0.225, 0.02, 0.625])  # Position for the colorbar
cbar =fig.colorbar(im, cax=cbar_ax)
cbar.set_ticks([0, 0.5, 1])

# Adjust layout to prevent overlap
# plt.tight_layout()

# axs[0].set_ylabel(r'Number of iterations')
# axs.set_ylabel(r'Normalized Loss')
# # uu = 5*20
# # ll = 15*20
# axs[0].set_xlim(5*20, 10*20)
# axs[1].set_xlim(5*20, 10*20)
#
# axs.set_ylim(0.6, 1.05)
# # axs[2].set_xlim(uu, ll)
# axs.set_ylim(0, 200)
# axs.set_xlim(50, 700)
# axs[1].set_ylim(0, 10)
# axs[0].set_yticks([0,2,4,6,8,10])
# axs.set_xticks(100*np.arange(8))
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')


# Add a legend
# fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.2, 0.3, 0.2, 0.1),ncol = 1 ,borderaxespad=0.1,handlelength=1.5,fancybox=True, framealpha=1)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()