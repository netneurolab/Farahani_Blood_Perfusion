"""

This code converts the MNI arterial territory maps from:

    Liu, C.F., Hsu, J., Xu, X., et al. (2023). Digital 3D Brain MRI Arterial Territories Atlas. Sci Data, 10, 74. https://doi.org/10.1038/s41597-022-01923-0

into CIFTI files for use with the HCP format.

Note: Related to Fig.5b & Fig.7b.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
from IPython import get_ipython
from globals import path_vessel, path_vessel_raw, path_wb_command, path_fsLR, path_templates

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
#------------------------------------------------------------------------------

#name = 'TerritoryVoxels_BMM'
name = 'ArterialAtlas_level2'
#name = 'ArterialAtlas'
#name = 'BorderZone_ProbAve'
#name = 'ProbArterialAtlas_average'
#name = 'ProbArterialAtlas_BMM'

command = path_wb_command + 'wb_command -volume-resample ' + \
      os.path.join(path_vessel_raw, name +'.nii') + \
      ' ' + path_fsLR + 'MNI152_T2_2mm.nii.gz '  + \
      'ENCLOSING_VOXEL '+ path_vessel + name +'_2mm.nii -affine ' + path_fsLR + 'affine.txt'
os.system(command)

command = path_wb_command + 'wb_command -volume-to-surface-mapping ' + \
    os.path.join(path_vessel_raw, name + '.nii') + ' ' + path_fsLR + 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii ' + \
    ' ' + path_vessel + name + '_R.func.gii -enclosing'
os.system(command)
command = path_wb_command + 'wb_command -volume-to-surface-mapping ' + \
    os.path.join(path_vessel_raw, name + '.nii') + ' ' + path_fsLR + 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii ' + \
    ' ' + path_vessel + name + '_L.func.gii -enclosing'
os.system(command)

command = path_wb_command + 'wb_command -cifti-create-dense-from-template ' + \
    path_templates + 'cortex_subcortex.dscalar.nii' + \
    ' ' + path_vessel + name + '.dscalar.nii' + \
    ' -volume-all ' +  path_vessel + name + '_2mm.nii' + \
    ' -metric CORTEX_LEFT ' + path_vessel + name + '_L.func.gii ' + \
    ' -metric CORTEX_RIGHT ' + path_vessel + name + '_R.func.gii '
os.system(command)

#------------------------------------------------------------------------------
# END
