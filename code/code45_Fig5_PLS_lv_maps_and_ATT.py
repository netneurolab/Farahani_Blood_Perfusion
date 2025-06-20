"""

Comparison of arrival time PC 1 and sex-specific PLS results (shown in Fig.5e).

female:

    r = 0.5193420141134194
    p = 0.000999000999000999

male:

    r = 0.4746296801317953
    p = 0.026973026973026972

Note: Related to Fig5.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from scipy.stats import pearsonr
from globals import path_results, path_figures
from numpy.polynomial.polynomial import Polynomial
from functions import vasa_null_Schaefer, pval_cal
from functions import convert_cifti_to_parcellated_SchaeferTian

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load ATT score map
#------------------------------------------------------------------------------

scores_data = np.load(path_results + 'arrival_PCscore.npy')[:, 0]
ATT = convert_cifti_to_parcellated_SchaeferTian(scores_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'X',
                                                      path_results,
                                                      'ATT_score_0')
pc_map = -1 * ATT.flatten() # make the sign more interpretable

#------------------------------------------------------------------------------
# Load the brain loading coming from PLS model
#------------------------------------------------------------------------------

pls_map = np.load(path_results + 'loadings_cortex_PLS_lv0_female.npy')
pls_map = pls_map.flatten()

#------------------------------------------------------------------------------
# Perform comparison of the two
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
'''
spins = vasa_null_Schaefer(nspins)
np.save(path_results + 'spins_watershed.npy', spins)
'''
spins =  np.load(path_results + 'spins_watershed.npy')
print(pearsonr(pc_map, pls_map))
r, generated_null = corr_spin(pls_map,
                             pc_map,
                             spins,
                             nspins)
p_value = pval_cal(r, generated_null, nspins)
print(p_value)

#------------------------------------------------------------------------------
# Plot the results as a scatterplot
#------------------------------------------------------------------------------

plt.figure(figsize = (5, 5))
plt.scatter(pls_map, pc_map,
            color = 'gray',
            s = 15)
p = Polynomial.fit(pls_map, pc_map, 1)
plt.plot(*p.linspace(),
         color = 'black',
         linewidth = 1)
plt.ylim(-54, 65)
plt.yticks(np.arange(-54, 66, 10))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(path_figures + 'arrival_time_and_pls_result_lv0_female.svg',
            format = 'svg')
plt.show()

#------------------------------------------------------------------------------
# END
