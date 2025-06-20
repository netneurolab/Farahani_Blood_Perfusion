"""

Similarity of disease maps and CBF/ATT score maps

Pearson correlation values (score CBF, score ATT (not inverted, raw)

LOAD   [-0.43464299,  0.32287539],
EOAD   [-0.12628344,  0.25548394],
3Rtau  [-0.3540689 ,  0.56461532],
4Rtau  [-0.25287647,  0.46717434],
TDP43A [-0.23709305,  0.55331567],
TDP43C [-0.61264791,  0.33744984],
PS1    [ 0.11650607,  0.13008755],
DLB    [-0.37233694,  0.20088804]])

Corrected p-spins (FDR):
    [0.003996  0.2443271  0.13786214  0.15984016  0.2443271  0.003996  0.32867133  0.00532801]

Note: Related to Fig.6.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import numpy as np
import seaborn as sns
import scipy.stats as stats
from functions import pval_cal
import matplotlib.pyplot as plt
from functions import vasa_null_Schaefer
from numpy.polynomial.polynomial import Polynomial
from statsmodels.stats.multitest import multipletests
from globals import path_results, path_disease, path_figures
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Load disease patterns
#------------------------------------------------------------------------------

nnodes = globals.nnodes_Schaefer
disease = ['LOAD', 'EOAD', '3Rtau', '4Rtau', 'TDP43A', 'TDP43C', 'PS1', 'DLB']
n_disease = len(disease)
disease_mapsminmax = np.zeros((n_disease, nnodes)) # (7, 400)
disease_maps = np.zeros((n_disease, nnodes)) # (7, 400)
for i in range(n_disease):
    disease_maps[i, :] = np.load(path_disease + 'mean_atropy_' + str(disease[i]) + '_Schaefer.npy').flatten()
    mask = np.isnan(disease_maps[i, :])
    disease_maps[i, mask] = np.nanmedian(disease_maps[i, :])
    min_val = np.min(disease_maps[i, :])
    max_val = np.max(disease_maps[i, :])
    disease_mapsminmax[i, :] = (disease_maps[i, :] - min_val) / (max_val - min_val)
disease_maps = disease_mapsminmax

#------------------------------------------------------------------------------
# Create spins for significance test
#------------------------------------------------------------------------------

nspins = 1000
flag_null = 0
if flag_null == 0:
    nulls_index_vasa = vasa_null_Schaefer(nspins)
    np.save(path_results + 'spin_disease.npy', nulls_index_vasa)
else:
   nulls_index_vasa =  np.load(path_results + 'spin_disease.npy')

#------------------------------------------------------------------------------
# Load perfusion/arrival score map
#------------------------------------------------------------------------------

perfusion_data = np.load(path_results + 'perfusion_PCscore.npy')[:, 0]
perfusion = convert_cifti_to_parcellated_SchaeferTian(perfusion_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test')

arrival_data = np.load(path_results + 'arrival_PCscore.npy')[:, 0]
arrival = convert_cifti_to_parcellated_SchaeferTian(arrival_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test')

#------------------------------------------------------------------------------
# Correlation between score maps and disease map
#------------------------------------------------------------------------------

def corr_spin(x, y, spins, nspins):
    """
    Spin test - account for spatial autocorrelation
    """
    rho, _ = stats.pearsonr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
         null[i], _ = stats.pearsonr(x, y[spins[:, i]])
    return rho, null

pval = np.zeros((n_disease, 2))
r = np.zeros((n_disease, 2))
for l in range(n_disease):
    print(disease[l])
    print(stats.pearsonr(disease_maps[l,:].T, perfusion))
    print(stats.pearsonr(disease_maps[l,:].T, arrival))
    r[l,0], generated_null = corr_spin(disease_maps[l,:].T,
                                 perfusion,
                                 nulls_index_vasa,
                                 nspins)
    pval[l, 0] = pval_cal(r[l, 0], generated_null, nspins)

    r[l,1], generated_null = corr_spin(disease_maps[l, :].T,
                                 arrival,
                                 nulls_index_vasa,
                                 nspins)
    pval[l, 1] = pval_cal(r[l, 1], generated_null, nspins)

    plt.figure(figsize = (5, 5))
    plt.scatter(perfusion,
                disease_maps[l, :],
                color = 'gray',
                linewidth = 1,
                alpha = 0.7,
                s = 15)
    p = Polynomial.fit(perfusion.flatten(), disease_maps[l, :].flatten(), 1)
    plt.plot(*p.linspace(),
             color = 'black',
             linewidth = 1)
    plt.title(str(disease[l]))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path_figures + str(disease[l]) + '_perfusion.svg',
                format = 'svg')
    plt.show()

    plt.figure(figsize = (5, 5))
    plt.scatter(arrival,
                disease_maps[l, :],
                color = 'gray',
                linewidth = 1,
                alpha = 0.7,
                s = 15)
    p = Polynomial.fit(arrival.flatten(), disease_maps[l, :].flatten(), 1)
    plt.plot(*p.linspace(),
             color = 'black',
             linewidth = 1)
    plt.title(str(disease[l]))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(path_figures + str(disease[l]) + '_arrival.svg',
                format = 'svg')
    plt.show()

# Extract the correlation values for visualization
# pval[:, 0] is for perfusion and pval[:, 1] is for arrival

corr_pval_perfusion = [multipletests(pval[:, 0].flatten(), method = 'fdr_bh')][0][1]
print(corr_pval_perfusion)

plt.figure()
sns.heatmap(r, cmap = 'coolwarm',
            vmax = -1 * np.min(r),
            vmin = np.min(r))
plt.savefig(path_figures + 'corr_disease_heatmap.svg',
            format = 'svg')
plt.show()
#------------------------------------------------------------------------------
# END
