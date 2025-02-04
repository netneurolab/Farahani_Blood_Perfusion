"""

# Correlation of neuropeptide receptor maps with cerebral blood flow score map
The noted p-values are after fdr-correction.

    Receptor: EDNRA
      Correlation: 0.48714089575107894
      p_spin: 0.000999
    --------------------
    Receptor: EDNRB
      Correlation: -0.5082851319481012
      p_spin: 0.000999
    --------------------
    Receptor: VIPR1
      Correlation: 0.4228782609009725
      p_spin: 0.000999
    --------------------
    Receptor: VIPR2
      Correlation: 0.4925200299702502
      p_spin: 0.000999
    --------------------

The noted p-values are FDR-corrected.

Note: Related to Fig.3c.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import scipy.io
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from IPython import get_ipython
from nilearn.datasets import fetch_atlas_schaefer_2018
from globals import path_results, path_figures, path_genes
from statsmodels.stats.multitest import multipletests
from functions import convert_cifti_to_parcellated_SchaeferTian, vasa_null_Schaefer, pval_cal

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

no_filter         = 0
parc              = 'Schaefer400'
nnodes            = globals.nnodes_Schaefer

#------------------------------------------------------------------------------
# Load gene-related information
#------------------------------------------------------------------------------

parc_file_mni = fetch_atlas_schaefer_2018(n_rois = nnodes)['maps']
cortex = np.arange(nnodes)

expression_st = scipy.io.loadmat(path_genes + 'gene_coexpression_schaefer_400_filtered.mat')['gene_coexpression']
name_genes = scipy.io.loadmat(path_genes + 'names_genes_schaefer_400_filtered.mat')['names']

columns_name = name_genes
n_genes = len(expression_st.T)
corr_vals = np.zeros((n_genes, 1))

#------------------------------------------------------------------------------
# Load cerebral blood perfusion score map
#------------------------------------------------------------------------------

perfusion_PC1_map_voxelwsie = np.load(path_results + 'perfusion_PCscore_0.npy')
perfusion_PC1 = convert_cifti_to_parcellated_SchaeferTian(perfusion_PC1_map_voxelwsie.reshape(1, globals.num_vertices_voxels),
                                                        'cortex',
                                                        'X',
                                                         path_results,
                                                        'perfusion_PCscore_0')
perfusion_PC1 = perfusion_PC1.flatten()

#------------------------------------------------------------------------------
# Find the correspondence
#------------------------------------------------------------------------------

expression_st = np.array(expression_st)
for n in range(n_genes):
    corr_vals[n] = pearsonr(expression_st[:,n].flatten(), perfusion_PC1)[0]

# Create a dataFrame based on the results
df_corr_features = pd.DataFrame({
    'genes': columns_name.flatten(),
    'Correlation': corr_vals.flatten()
})

#------------------------------------------------------------------------------
# Examine specific genes of interest
#------------------------------------------------------------------------------

index = np.where((columns_name == 'EDNRA') |
                 (columns_name == 'EDNRB') |
                 (columns_name == 'VIPR1') |
                 (columns_name == 'VIPR2')
                 )[1]
num_genes = len(index)
expression_st = np.array(expression_st)
corr_values = np.zeros((num_genes, 1))
p_values = np.zeros((num_genes, 1))

#------------------------------------------------------------------------------
# Statistical test - spin test
#------------------------------------------------------------------------------

nspins = 1000
spins = vasa_null_Schaefer(nspins)
def corr_spin(x, y, spins, nspins):
    """
    Spin test - account for spatial autocorrelation
    """
    rho, _ = pearsonr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
         null[i], _ = pearsonr(x, y[spins[:, i]])
    return rho, null


for n in range(num_genes):
    data = expression_st[:,index[n]]
    receptor_data = data.flatten()
    corr_values[n, 0], _ = (pearsonr(perfusion_PC1, receptor_data))
    print('correlation value is:' + str(corr_values[n, 0]))

    r, generated_null = corr_spin(receptor_data,
                                 perfusion_PC1,
                                 spins,
                                 nspins)
    p_values[n, 0] = pval_cal(r, generated_null, nspins)
    print(p_values[n, 0] )
    print('------------------')

# Apply FDR correction to the array of p-values
_, corrected_p_values, _, _ = multipletests(p_values.flatten(), method = 'fdr_bh')
corrected_p_values = corrected_p_values.reshape(p_values.shape)
print("Corrected p-values:", corrected_p_values)

#------------------------------------------------------------------------------
# Show the results as a heatmap
#------------------------------------------------------------------------------

plt.figure(figsize = (5 ,5))
sns.heatmap(corr_values, vmin = -0.61, vmax = 0.61, cmap = 'coolwarm')
plt.tight_layout()
plt.savefig(path_figures + 'peptides.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END