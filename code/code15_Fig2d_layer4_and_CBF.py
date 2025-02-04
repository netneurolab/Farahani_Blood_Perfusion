"""

# Assess the correspondence between CBF and cortical Layer IV

r      = 0.5247664730349205
p_spin = 0.002997002997002997

Note: Related to Fig.1d.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr
from numpy.polynomial.polynomial import Polynomial
from functions import vasa_null_Schaefer, pval_cal
from functions import convert_cifti_to_parcellated_SchaeferTian
from globals import path_results, path_genes, path_figures

#------------------------------------------------------------------------------
# Load gene data
#------------------------------------------------------------------------------

path_gene_maps_abagen = os.path.join(path_genes, 'names_genes_schaefer_400_filtered.mat')
gene_maps_abagen_raw = loadmat(path_gene_maps_abagen)['names']

gene_maps_abagen = gene_maps_abagen_raw.ravel()

if isinstance(gene_maps_abagen[0], bytes):
    gene_maps_abagen = [g.decode('utf-8') for g in gene_maps_abagen]
else:
    gene_maps_abagen = gene_maps_abagen.tolist()
print(f"Number of genes in Abagen data: {len(gene_maps_abagen)}")

path_data_abagen = os.path.join(path_genes, 'gene_coexpression_schaefer_400_filtered.mat')
data_abagen = loadmat(path_data_abagen)['gene_coexpression']
print("data_abagen shape:", data_abagen.shape)

#------------------------------------------------------------------------------
# layer IV genes
#------------------------------------------------------------------------------

cleaned_gene_list = {'COL6A1', 'CUX2', 'TRMT9B', 'GRIK4', 'RORB'}

common_genes = [g for g in cleaned_gene_list if g in gene_maps_abagen]
print(f"Found {len(common_genes)} matching genes in Abagen data out of {len(cleaned_gene_list)} requested.")

idx_common = [gene_maps_abagen.index(g) for g in common_genes]
layer_IV = data_abagen[:, idx_common]
print("layer_IV gene shape:", layer_IV.shape)

#------------------------------------------------------------------------------
# Normalize the gene maps and calculate the mean of all five genes - refer to Burt 2018 paper
#------------------------------------------------------------------------------

zscored_layer_IV = zscore(layer_IV, axis = 0)
mean_layer_IV = np.mean(zscored_layer_IV, axis = 1)

#------------------------------------------------------------------------------
# Now compare this pattern with CBF and compute the p-spin value.
#------------------------------------------------------------------------------

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

nspins = 1000
spins = vasa_null_Schaefer(nspins)
np.save(path_results + 'spins_layers.npy', spins)

scores_data = np.load(path_results + 'perfusion_PCscore.npy')[:,0]
perfusion = convert_cifti_to_parcellated_SchaeferTian(scores_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test')
print(pearsonr(mean_layer_IV, perfusion))

r, generated_null = corr_spin(mean_layer_IV,
                             perfusion,
                             spins,
                             nspins)
p_value = pval_cal(r, generated_null, nspins)
print(p_value)
print('------------------')

#------------------------------------------------------------------------------
# Plot the results
#------------------------------------------------------------------------------

def plot_significant_scatter(x, y, xlabel, ylabel, title, file_name):
    plt.figure(figsize = (5, 5))
    plt.scatter(x, y,
                color = 'gray',
                s = 15)
    p = Polynomial.fit(x, y, 1)
    plt.plot(*p.linspace(),
             color = 'black',
             linewidth = 1)
    plt.ylim(-56, 66)

    plt.yticks(np.arange(-56, 66, 10))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_name,
                format = 'svg')
    plt.show()

if p_value < 0.05:
    print(pearsonr(mean_layer_IV, perfusion.flatten()))
    plot_significant_scatter(mean_layer_IV.flatten(),
                             perfusion.flatten(),
                             'Mean_layer_IV',
                             'Mean Perfusion',
                             ' Mean_layer_IV vs. Mean Perfusion',
                             path_figures + 'Mean_layer_IV_vs_perfusion_cortical.svg')

#------------------------------------------------------------------------------
# END