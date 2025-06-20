"""

# rev 1: What is the relationship between blood perfusion and other cortical layers?

Results from current analysis:
    
- **Layer III (1–3):**
    r = -0.1335
    p_spin = 0.4426  → Not significant

- **Layer V–VI (5–6):**
    r = -0.1338
    p_spin = 0.5564  → Not significant

- **Layer IV:**
    r = 0.5248
    p_spin = 0.00299  → Significant association

We also report which genes were missing from the Abagen dataset for transparency.

# NOTE: Related to Fig.S8.

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
from globals import path_results, path_genes, path_figures
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import vasa_null_Schaefer, pval_cal, save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# Load gene data
#------------------------------------------------------------------------------

layer = 'IV'
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
# layer specific genes
#------------------------------------------------------------------------------

if layer == 'III':
    cleaned_gene_list = {'C1QL2', 'C20orf103', 'CARTPT', 'DISC1', 'GLRA3',
                        'GSG1L', 'IGSF11', 'INPP4B', 'MFGE8', 'PVRL3', 
                        'RASGRF2','SV2C','WFS1'}
elif layer == 'IV':
    cleaned_gene_list = {'COL6A1', 'CUX2', 'TRMT9B', 'GRIK4', 'RORB'}
elif layer == 'V':
    cleaned_gene_list = {'ADRA2A', 'AKR1C3', 'ANXA1', 'B3GALT2', 'CDH24',
                         'CTGF', 'ETV1', 'FAM3C', 'FOXP2', 'HTR2C',
                         'KCNK2', 'NPY2R', 'NR4A2', 'NTNG2', 'OPRK1',
                         'PCDH17', 'PCDH20', 'PCP4', 'PDE1A', 'RPRM',
                         'RXFP1', 'SNTB1', 'SYT10', 'SYT6', 'TLE4',
                         'TOX', 'TRIB2', 'VAT1L'}

# Check for missing genes
cleaned_gene_list = list(cleaned_gene_list)
common_genes = [g for g in cleaned_gene_list if g in gene_maps_abagen]
missing_genes = [g for g in cleaned_gene_list if g not in gene_maps_abagen]

print(f"✅ Found {len(common_genes)} matching genes in Abagen data out of {len(cleaned_gene_list)} requested.")
print("🧬 Matching genes:", common_genes)
if missing_genes:
    print(f"❌ Missing genes ({len(missing_genes)}):", missing_genes)
else:
    print("✅ No missing genes.")

idx_common = [gene_maps_abagen.index(g) for g in common_genes]
layer_data = data_abagen[:, idx_common]
print("layer_" + layer + "gene shape:", layer_data.shape)

#------------------------------------------------------------------------------
# Normalize the gene maps and calculate the mean of all genes - refer to Burt 2018 paper
#------------------------------------------------------------------------------

zscored_layer_data = zscore(layer_data, axis = 0)
mean_layer_data = np.mean(zscored_layer_data, axis = 1)
save_parcellated_data_in_SchaeferTian_forVis(mean_layer_data,
                                             'cortex',
                                             'S1',
                                             path_results,
                                             layer +'_rev1')

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
#spins = vasa_null_Schaefer(nspins)
spins = np.load(path_results + 'spins_layers.npy')

scores_data = np.load(path_results + 'perfusion_PCscore.npy')[:,0]
perfusion = convert_cifti_to_parcellated_SchaeferTian(scores_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test')
print(pearsonr(mean_layer_data, perfusion))

r, generated_null = corr_spin(mean_layer_data,
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

print(pearsonr(mean_layer_data, perfusion.flatten()))
plot_significant_scatter(mean_layer_data.flatten(),
                         perfusion.flatten(),
                         'Mean_layer',
                         'Mean Perfusion',
                         'Mean_layer vs. Mean Perfusion',
                         path_figures + 'Mean_layer_' + layer + '_vs_perfusion_cortical.svg')

#------------------------------------------------------------------------------
# END
