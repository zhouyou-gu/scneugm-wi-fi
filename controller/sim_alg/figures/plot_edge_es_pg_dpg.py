import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 400
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
fig, axs = plt.subplots(3, 1)
# fig_width_in = (fig_width_px * 3 + 40) / dpi  # Adjust width for three subplots and spacing
fig.set_size_inches(fig_width_in, fig_height_in)

# Define the paths to your three data files
folder = [
    os.path.join(get_controller_path(), "sim_alg/train_and_test_es_vs_pg_dpg/log-train_es/train_es-2024-December-18-16-23-40-ail/ES_GGM.edge_value_raw.final.txt"),
    os.path.join(get_controller_path(), "sim_alg/train_and_test_es_vs_pg_dpg/log-train_pg/train_pg-2024-December-18-16-19-42-ail/PG_GGM.edge_value_raw.final.txt"),
    os.path.join(get_controller_path(), "sim_alg/train_and_test_es_vs_pg_dpg/log-train_dpg/train_dpg-2024-December-18-16-27-56-ail/PG_GGM.edge_value_raw.final.txt"),
]

# Initialize a list to hold your distribution arrays
distribution_arrays = []

# Define bins for the distributions
num_bins = 10
bins = np.linspace(0, 1, num_bins + 1)  # e.g., [0.0, 0.1, 0.2, ..., 1.0]
bin_centers = (bins[:-1] + bins[1:]) / 2  # For labeling if needed

# Load and process the data
for idx, data_file in enumerate(folder):
    # Check if the file exists
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Load the data
    data = np.genfromtxt(data_file, delimiter=',')[:,3:]
    
    # Initialize an array to hold distribution counts
    # Shape: (rows, num_bins)
    distributions = np.zeros((data.shape[0], num_bins))
    
    # Compute the histogram for each row
    for row_idx in range(data.shape[0]):
        counts, _ = np.histogram(data[row_idx], bins=bins)
        distributions[row_idx] = counts
    
    distribution_arrays.append(distributions.T)

# Compute global vmax across all distributions for consistent color scaling
global_vmax = max(dist.max() for dist in distribution_arrays)

# Plot the distribution counts as heatmaps
for i in range(3):
    im = axs[i].imshow(distribution_arrays[i], 
                       cmap='viridis', 
                       aspect='auto', 
                       vmin=0, 
                       vmax=global_vmax)
    axs[i].set_title(data_name_list[i])
    axs[i].set_xlabel('Value Bins')


    # Get the number of rows and columns
    num_rows, num_cols = distribution_arrays[i].shape

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
    
    # axs[i].set_position([0.125+i*0.25, 0.225, 0.225, 0.625])

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