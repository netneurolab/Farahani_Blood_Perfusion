"""

Create semantic map for vessels

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import warnings
import numpy as np
import nibabel as nib
from IPython import get_ipython
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore
from statsmodels.stats.multitest import multipletests
from globals import path_results, path_figures, path_neurosynth
from functions import convert_cifti_to_parcellated_SchaeferTian, pval_cal

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load neurosynth data - cortical only
#------------------------------------------------------------------------------

neurosynth_data = np.load(path_neurosynth + 'neurosynth_data_S4.npy')[globals.nnodes_S4:,:]
neurosynth_columns = np.load(path_neurosynth + 'neurosynth_names.npy')
num_neurosynth_terms = len(neurosynth_columns)

neurosynth_data = zscore(neurosynth_data, axis = 0)

#------------------------------------------------------------------------------
# Load gradient maps
#------------------------------------------------------------------------------

gradients = np.zeros((globals.nnodes_Schaefer, 2))

gradient_temp = os.path.join(path_results, 'gradient_0_perfusion.dscalar.nii')
gradient_temp_data = nib.cifti2.load(gradient_temp).get_data()
gradients[:, 0] = convert_cifti_to_parcellated_SchaeferTian(gradient_temp_data.reshape(1, globals.num_vertices_voxels),
                                          'cortex',
                                          'S1',
                                          path_results,
                                          'G0').flatten()
gradient_temp = os.path.join(path_results, 'gradient_1_perfusion.dscalar.nii')
gradient_temp_data = nib.cifti2.load(gradient_temp).get_data()
gradients[:, 1] = (convert_cifti_to_parcellated_SchaeferTian(gradient_temp_data.reshape(1, globals.num_vertices_voxels),
                                          'cortex',
                                          'S1',
                                          path_results,
                                          'G1').flatten())

#------------------------------------------------------------------------------
# Calculate correlation values
#------------------------------------------------------------------------------

def corr_spin(x, y, nulls_matrix, nspins):
    """
    Spin test - account for spatial autocorrelation
    """
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))
    # null correlation
    for i in range(nspins):
         null[i], _ = pearsonr(x, y[nulls_matrix[:, i]])
    return rho, null

nspins = 1000
spin_res = np.load(path_results + 'spin_metabolism.npy')

p_value = np.zeros((num_neurosynth_terms, 2))
r_real = np.zeros((num_neurosynth_terms, 2))

for i in range(2):
    for n in range(num_neurosynth_terms):
        data = neurosynth_data[:, n].flatten()
        # Replace NaN values with the median
        nan_indices = np.where(np.isnan(data))
        median_value = np.nanmedian(data)
        data[nan_indices] = median_value
        r_real[n, i], _ = pearsonr(gradients[:, i].flatten(), data.flatten())
        r, gen_nul = corr_spin(data.flatten(),
                             gradients[:, i].flatten(),
                             spin_res,
                             nspins)
        p_value[n, i] = pval_cal(r, gen_nul, nspins)
        print(n)

#------------------------------------------------------------------------------
# Visualization
#------------------------------------------------------------------------------

significance_level = 0.05

# Flatten the p-value array to apply FDR correction on all values at once
p_values_flat = p_value.flatten()

# Apply FDR correction
_, corrected_p_values_flat, _, _ = multipletests(p_values_flat, method = 'fdr_bh')

# Reshape back to the original shape
corrected_p_values = corrected_p_values_flat.reshape(p_value.shape)

# Now corrected_p_values can be used for filtering
# Colors and labels for each significance condition
colors = []
color_labels = []
for i in range(num_neurosynth_terms):
    # Check for significance in each dimension using corrected p-values
    p0_significant = p_value[i, 0] < significance_level
    p1_significant = p_value[i, 1] < significance_level
    notsig = (p_value[i, 0] >= significance_level) & (p_value[i, 1] >= significance_level)

    if p0_significant and p1_significant:
        colors.append('k') # Black for all significant
        color_labels.append('All significant (black)')
    elif p0_significant:
        colors.append('r') # Red for significant in G0 only
        color_labels.append('G0 significant (red)')
    elif p1_significant:
        colors.append('g') # Green for significant in G1 only
        color_labels.append('G1 significant (green)')
    elif notsig:
        colors.append('silver') # Silver for non-significant in both
        color_labels.append('Non-significant (silver)')

# Create a 2D scatter plot
fig = plt.figure(figsize = (15, 15))
ax = fig.add_subplot(111)

# Plot all points in 2D space with assigned colors
scatter = ax.scatter(r_real[:, 0],
                     r_real[:, 1],
                     color = colors,
                     s = 100)

# Annotate each point with the term and its correlation values
for i, term in enumerate(neurosynth_columns):
    r_values = f"({r_real[i, 0]:.2f}, {r_real[i, 1]:.2f})"
    ax.text(r_real[i, 0],
            r_real[i, 1],
            f"{term}",
            size = 10,
            ha = 'center')
ax.set_xlabel('G1 Correlation', fontsize = 14)
ax.set_ylabel('G2 Correlation', fontsize = 14)
unique_colors = list(set(zip(colors, color_labels)))
handles = [plt.Line2D([0], [0],
                      marker = 'o',
                      color = 'w',
                      label = label,
                      markerfacecolor = color,
                      markersize = 10)
           for color, label in unique_colors]
ax.legend(handles=handles,
          loc = 'upper right',
          title = "Significance Legend")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the plot as an SVG file
plt.tight_layout()
plt.savefig(path_figures + 'neurosynth_gradients_stroke_with_nonsig.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END