"""

Correlate the beta map (HCP-A) and the arrival time score map and color dots based on an ATLAS

r = -0.711367319189178
p_spin = 0.000999000999000999

Note: Related to Fig.5c.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from IPython import get_ipython
from scipy.stats import pearsonr
import matplotlib.colors as mcolors
from functions import vasa_null_Schaefer
from numpy.polynomial.polynomial import Polynomial
from globals import path_results, path_figures, path_vessel
from functions import save_parcellated_data_in_SchaeferTian_forVis
from functions import convert_cifti_to_parcellated_SchaeferTian, pval_cal

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load data (vertex-wise) - beta coeff of perfusion in HCP-A
#------------------------------------------------------------------------------

data = np.load(path_results + 'Aging_betas_age_perfusion.npy')
data = data.flatten()

beta_age = np.load(path_results + 'Aging_betas_age_perfusion.npy')
beta_age = convert_cifti_to_parcellated_SchaeferTian(beta_age.reshape(1, globals.num_vertices_voxels),
                                                        'cortex',
                                                        'X',
                                                         path_results,
                                                        'Aging_betas_age_perfusion')
beta_age = beta_age.flatten()

#------------------------------------------------------------------------------
# Get ATT score map
#------------------------------------------------------------------------------

scores_data = np.load(path_results + 'arrival_PCscore.npy')[:, 0]
ATT = convert_cifti_to_parcellated_SchaeferTian(scores_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'X',
                                                      path_results,
                                                      'ATT_score_0')
pc_map = -1* ATT.flatten() # make the sign more interpretable

# Save the maps for future use
save_parcellated_data_in_SchaeferTian_forVis(pc_map,
                        'cortex',
                        'X',
                        path_results,
                        'ATT_score_0_inverted')

#------------------------------------------------------------------------------
# Compare CBF aging effect and ATT score map
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
np.save(path_results + 'spins_watershed.npy', spins)

spins =  np.load(path_results + 'spins_watershed.npy')
print(pearsonr(pc_map, beta_age))
r, generated_null = corr_spin(beta_age,
                             pc_map,
                             spins,
                             nspins)
p_value = pval_cal(r, generated_null, nspins)
print(p_value)

#------------------------------------------------------------------------------
# Load border zone from ATLAS
#------------------------------------------------------------------------------

border_zone = nib.load(path_vessel + 'ArterialAtlas_level2.dscalar.nii').get_fdata()
border_zone, _ = divmod(border_zone, 1)
border_zone_map = convert_cifti_to_parcellated_SchaeferTian(border_zone.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'border_zone')

border_zone_map[(border_zone_map == 1) | (border_zone_map == 2) |
                (border_zone_map == 3) | (border_zone_map == 4) |
                (border_zone_map == 5) | (border_zone_map == 6)] = 0

plt.hist(border_zone_map)
border_zone_map[border_zone_map > 1] = 1

save_parcellated_data_in_SchaeferTian_forVis(border_zone_map,
                        'cortex',
                        'X',
                        path_results,
                        'border_zone_map')

#------------------------------------------------------------------------------
# Plot the results
#------------------------------------------------------------------------------

# Create a custom two-color colormap: index 0 -> silver, index 1 -> red
cmap = mcolors.ListedColormap(['silver', 'red'])

plt.figure(figsize = (5,5))
scatter = plt.scatter(beta_age, pc_map,
                      c = border_zone_map,  # Use border_zone_map for colors
                      cmap = cmap,
                      alpha = 0.9,
                      s = 15,
                      vmin = 0,
                      vmax = 1)

p = Polynomial.fit(beta_age, pc_map, 1)
plt.plot(*p.linspace(),
         color = 'black',
         linewidth = 1)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'arrival_time_and_beta_age_colored.svg',
            format ='svg')
plt.show()

#------------------------------------------------------------------------------
# END