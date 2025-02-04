"""

Create the nulls for gene expression analysis

Note: Related to Fig.3a,b.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import globals
import warnings
import numpy as np
from scipy.io import savemat
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import vasa_null_Schaefer
from globals import path_results

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Constants
#------------------------------------------------------------------------------

nspins = 50000

#------------------------------------------------------------------------------
 #Parcellate cerebral perfusion score map
#------------------------------------------------------------------------------

scores_data = np.load(path_results + 'perfusion_PCscore.npy')[:, 0]
perfusion_PC1 = convert_cifti_to_parcellated_SchaeferTian(scores_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test')

data_to_save = {'perfusion': perfusion_PC1}
savemat(path_results + 'perfusion1_pc0_map_cortical_matlab.mat', data_to_save)

#------------------------------------------------------------------------------
# Vasa spins
#------------------------------------------------------------------------------

nulls_index_vasa = vasa_null_Schaefer(nspins)
spin_res = []
for i in range(nspins):
    spin_res.append(perfusion_PC1[nulls_index_vasa[:, i], 0])

spin_res = np.array(spin_res)
data_to_save = {'spin_res' : spin_res}
savemat(path_results + 'perfusion1_pc0_vasa_cortical_matlab.mat', data_to_save)

#------------------------------------------------------------------------------
# END