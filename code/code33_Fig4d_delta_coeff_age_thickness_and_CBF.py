"""

Comparison of age-beta maps of CBF and cortical thickness (HCP-D)

pearson correlation voxel-wise maps:
    (0.25143252161631074, 0.0)

pearson correlation parcel-wise maps:
    (0.3334271484680586, 7.673694961680693e-12)

p-spin (parcel-wise):
    0.003996003996003996

Note: Related to Fig.4d.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
from scipy.stats import pearsonr
from IPython import get_ipython
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
from functions import vasa_null_Schaefer, pval_cal
from globals import path_results, path_figures, path_yeo
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load age-effect maps
#------------------------------------------------------------------------------

thickness_beta = np.load(path_results + 'Dev_betas_age_thickness.npy')
perfusion_beta = np.load(path_results + 'Dev_betas_age_perfusion.npy')[:globals.num_cort_vertices_noMW]

#------------------------------------------------------------------------------
# Parcellate data
#------------------------------------------------------------------------------

perfusion_beta_p = convert_cifti_to_parcellated_SchaeferTian(perfusion_beta.reshape(1, globals.num_cort_vertices_noMW),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test').flatten()
thickness_beta_p = convert_cifti_to_parcellated_SchaeferTian(thickness_beta.reshape(1, globals.num_cort_vertices_noMW),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test').flatten()

#------------------------------------------------------------------------------
# Plot the association between the two - vertex-wise
#------------------------------------------------------------------------------

# Plot scatter plot
plt.figure(figsize = (5, 5))
plt.scatter(thickness_beta, perfusion_beta,
            color =  'silver', alpha = 0.7)
plt.savefig(path_figures + 'dev_thickness_perfusion_CBF_age_effect_vertex_wise.svg',
            bbox_inches = 'tight', dpi = 300)
plt.show()
print(pearsonr(thickness_beta, perfusion_beta))

#------------------------------------------------------------------------------
# Parcel-wise comparison
#------------------------------------------------------------------------------

atlas_7Network = np.load(path_yeo + "Schaefer2018_400Parcels_7Networks.npy")

c_unimodal = [67/255, 160/255, 71/255]     # Green for unimodal regions
c_multimodal = [255/255, 140/255, 0/255]   # Orange for transmodal regions

multimodal = np.where((atlas_7Network == 6) | (atlas_7Network == 5))[0]
unimodal = np.where((atlas_7Network == 0) | (atlas_7Network == 1))[0]
between = np.where((atlas_7Network == 2) | (atlas_7Network == 3) | (atlas_7Network == 4))[0]

# Parcel-wise scatter plot with region coloring
plt.figure(figsize=(5, 5))
plt.scatter(thickness_beta_p[unimodal],
            perfusion_beta_p[unimodal],
            color = c_unimodal,
            alpha = 0.7,
            s = 15,
            label = 'Unimodal')
plt.scatter(thickness_beta_p[multimodal],
            perfusion_beta_p[multimodal],
            color = c_multimodal,
            alpha = 0.7,
            s = 15,
            label = 'Multimodal')
plt.scatter(thickness_beta_p[between],
            perfusion_beta_p[between],
            color = 'silver',
            alpha = 0.7,
            s = 15,
            label = 'In between')
b, m = polyfit(thickness_beta_p, perfusion_beta_p, 1)
plt.plot(thickness_beta_p, b + m * thickness_beta_p,
         color = 'black', linewidth = 1)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('thickness and CBF')
plt.tight_layout()
#plt.xlabel('Thickness Beta')
#plt.ylabel('Perfusion Beta')
plt.savefig(path_figures + 'dev_thickness_perfusion_CBF_age_effect_parcellated_colored.svg',
            bbox_inches = 'tight', dpi = 300)

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
print(pearsonr(thickness_beta_p, perfusion_beta_p))
r, generated_null = corr_spin(thickness_beta_p,
                             perfusion_beta_p,
                             spins,
                             nspins)
p_value = pval_cal(r, generated_null, nspins)
print(p_value)

#------------------------------------------------------------------------------
# END
