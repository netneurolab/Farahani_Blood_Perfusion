"""

# Correlation of neurotransmitter receptor maps with cerebral blood flow score map.
The reported p-values are FDR-corrected.

    Receptor: 5HT1a_cumi_hc8_beliveau
      Correlation: -0.609
      p_spin: 0.00100
      p_corr: 0.0034965
    --------------------
    Receptor: 5HT1b_p943_hc65_gallezot
      Correlation: 0.418
      p_spin: 0.00100
      p_corr: 0.0034965
    --------------------
    Receptor: 5HT2a_cimbi_hc29_beliveau
      Correlation: 0.203
      p_spin: 0.19481
      p_corr: 0.25524476
    --------------------
    Receptor: 5HT4_sb20_hc59_beliveau
      Correlation: -0.203
      p_spin: 0.21878
      p_corr: 0.25524476
    --------------------
    Receptor: 5HT6_gsk_hc30_radhakrishnan
      Correlation: -0.033
      p_spin: 0.68931
      p_corr: 0.68931069
    --------------------
    Receptor: GABAa-bz_flumazenil_hc16_norgaard
      Correlation: 0.324
      p_spin: 0.00200
      p_corr: 0.004662
    --------------------
    Receptor: NMDA_ge179_hc29_galovic
      Correlation: -0.183
      p_spin: 0.07493
      p_corr: 0.13111888
    --------------------

Note: Related to Fig.3c.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

# Local imports
from globals import (
    path_results, path_figures, path_receptors,
    num_vertices_voxels
)
from functions import (
    pval_cal, convert_cifti_to_parcellated_SchaeferTian, vasa_null_Schaefer
)

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load receptor data from CSV
#------------------------------------------------------------------------------

df = pd.read_csv(path_receptors + 'receptors_ctx.csv')

# Columns except "labels" are assumed to be receptor columns
receptor_cols = [col for col in df.columns if col != "labels"]
receptor_names = receptor_cols
receptor_data = df[receptor_cols].values  # Shape: (400, # of receptors)
num_receptors = len(receptor_names)

# Replace any NaN values in each receptor column with the column mean
for n in range(num_receptors):
    col_values = receptor_data[:, n]
    if np.isnan(col_values).any():
        mean_value = np.nanmean(col_values)
        col_values = np.where(np.isnan(col_values), mean_value, col_values)
        receptor_data[:, n] = col_values

#------------------------------------------------------------------------------
# Load cerebral blood perfusion score map
#------------------------------------------------------------------------------

perfusion_PC1_map_voxelwise = np.load(path_results + 'perfusion_PCscore_0.npy')
perfusion_PC1 = convert_cifti_to_parcellated_SchaeferTian(perfusion_PC1_map_voxelwise.reshape(1, num_vertices_voxels),
                                                            'cortex',
                                                            'X',
                                                            path_results,
                                                            'perfusion_PCscore_0').flatten()

#------------------------------------------------------------------------------
# Generate spins
#------------------------------------------------------------------------------

nspins = 1000
spins = vasa_null_Schaefer(nspins)

def corr_spin(x, y, spins, nspins):
    """
    Perform a spin test to account for spatial autocorrelation.
    """
    rho, _ = pearsonr(x, y)
    null = np.zeros(nspins)
    for i in range(nspins):
        null[i], _ = pearsonr(x, y[spins[:, i]])
    return rho, null

#------------------------------------------------------------------------------
# Correlate perfusion (PC1) with each receptor, apply spin test
#------------------------------------------------------------------------------

real_r = np.zeros((num_receptors, 1))
p_value = np.zeros((num_receptors, 1))

for n in range(num_receptors):
    receptor_vector = receptor_data[:, n]
    # Pearson correlation
    corr_val, _ = pearsonr(perfusion_PC1, receptor_vector)
    real_r[n, 0] = corr_val
    print(f"Receptor: {receptor_names[n]}")
    print(f"  Correlation: {corr_val:.3f}")

    # Spin test
    r, nulls = corr_spin(receptor_vector, perfusion_PC1, spins, nspins)
    p_value[n, 0] = pval_cal(r, nulls, nspins)
    print(f"  p_spin: {p_value[n, 0]:.5f}")
    print('-' * 20)

# Correct for multiple comparisons using FDR
_, p_value_corrected, _, _ = multipletests(p_value.flatten(),
                                           method = 'fdr_bh')

#------------------------------------------------------------------------------
# Plot the results as a heatmap
#------------------------------------------------------------------------------

plt.figure(figsize=(5, 5))
sns.heatmap(real_r, vmin = -0.61, vmax = 0.61, cmap = 'coolwarm',
            yticklabels=receptor_names)
plt.title("Correlations (Perfusion PC1 vs. Receptors)")
plt.tight_layout()
plt.savefig(path_figures + 'receptors.svg', format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END
