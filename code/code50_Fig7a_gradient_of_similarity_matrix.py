"""

Use vertex-wise data and build the gradients of inter-individual similarity matrix

Results are saved as "{name}_gradient.dscalar.nii"

Lambda values:
    11.5485
    7.05128
    4.1643
    2.14665
    2.05995

Note: Related Fig.7a.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import warnings
import scipy.io
import numpy as np
import nibabel as nib
import seaborn as sns
from scipy.stats import zscore
import matplotlib.pyplot as plt
from IPython import get_ipython
from neuromaps import transforms
from brainspace.gradient import GradientMaps
from nilearn.connectome import ConnectivityMeasure
from globals import path_results, path_templates, path_fsLR, path_wb_command, path_medialwall, path_figures

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load medial-wall information
#------------------------------------------------------------------------------

mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask']
mask_medial_wall = mask_medial_wall.astype(np.float32)

#------------------------------------------------------------------------------
# Some needed templates to use later on
#------------------------------------------------------------------------------

# arrival or perfusion
name = 'perfusion'

Left = transforms.fslr_to_fslr(path_results + name + '_cortical_32k_L.func.gii',
                               target_density ='4k',
                               hemi = {'L'},
                               method ='linear')

Right = transforms.fslr_to_fslr(path_results +  name + '_cortical_32k_R.func.gii',
                               target_density ='4k',
                               hemi = {'R'},
                               method ='linear')

template_paths = {'cortex': os.path.join(path_templates, 'cortex.dscalar.nii')}
templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}
template = templates['cortex']

#------------------------------------------------------------------------------
# Load cortical and subcortical data - low res
#------------------------------------------------------------------------------

dev_all_data_cortex = np.load(path_results + '4k_'+ name +'_all_subjects_dev.npy')
dev_all_data_subortex = np.load(path_results + '4k_subcortex_'+ name +'_all_subjects_dev.npy')
dev_all_data = np.concatenate((dev_all_data_cortex, dev_all_data_subortex))

age_all_data_cortex = np.load(path_results + '4k_' + name + '_all_subjects_aging.npy')
age_all_data_subortex = np.load(path_results + '4k_subcortex_' + name + '_all_subjects_aging.npy')
age_all_data = np.concatenate((age_all_data_cortex, age_all_data_subortex))

all_data = np.concatenate((dev_all_data, age_all_data), axis = 1)
np.save(path_results + name + '_4k_all.npy', all_data)

#------------------------------------------------------------------------------
# Perform gradient analysis on the data
#------------------------------------------------------------------------------

num_components_g = 5
all_data = zscore(all_data, axis = 0)
data_normalized = zscore(all_data, axis = 1)

# Find rows (parcels) with NaNs or Infs
problematic_parcels = np.any(np.isnan(data_normalized) | np.isinf(data_normalized), axis = 1)
data_normalized_clean = data_normalized[~problematic_parcels]

correlation_measure = ConnectivityMeasure(kind = 'covariance')
corr_similarity = correlation_measure.fit_transform([data_normalized_clean.T])[0]

gm = GradientMaps(n_components = num_components_g,
                  approach = 'dm',
                  random_state = 1234)
gm.fit(corr_similarity)
gradients = gm.gradients_
gradients = np.float64(gradients * 1)

# Plot the eigenvalues
plt.figure(figsize = (3, 5))
plt.scatter(range(5), gm.lambdas_, color = 'silver')
plt.tight_layout()
plt.savefig(path_figures + 'gradient_lambdas.svg',
            format ='svg')
plt.show()

# Reinsert NaNs in the original parcel locations
full_gradients = np.full((all_data.shape[0], num_components_g), np.nan)
full_gradients[~problematic_parcels] = gradients

'''
for i in range(3):
    ordered_sim = corr_similarity[np.argsort(gradients[:,i]),:][:,np.argsort(gradients[:,i])]
    plt.figure(figsize = (5, 5))
    sns.heatmap(ordered_sim, vmin = -0.5, vmax = 0.5, cmap = 'coolwarm')
    plt.tight_layout()
    plt.savefig(path_figures + 'similarity_matrix_sorted_based_on_gradient_' + name + '_' + str(i) + '.png',
                format ='png')
    plt.show()
'''
#------------------------------------------------------------------------------
# Save cortical part
#------------------------------------------------------------------------------

for num_gradient in range(num_components_g):
    Left[0].darrays[0].data = full_gradients[:4002, num_gradient]
    Right[0].darrays[0].data = full_gradients[4002:8004, num_gradient]
    Left_all_back = transforms.fslr_to_fslr(Left,
                                            target_density = '32k',
                                            hemi = {'L'},
                                            method = 'linear')

    Right_all_back = transforms.fslr_to_fslr(Right,
                                            target_density = '32k',
                                            hemi = {'R'},
                                            method = 'linear')
    tosaveL = Left_all_back[0].darrays[0].data
    tosaveR = Right_all_back[0].darrays[0].data
    tosave = np.concatenate((tosaveL, tosaveR))
    tosave59k = tosave[(mask_medial_wall == 1).flatten()]
    new_img = nib.Cifti2Image(tosave59k.reshape((1, -1)),
                              header = template.header,
                              nifti_header = template.nifti_header)
    new_img.to_filename(os.path.join(path_results, 'gradient_' + str(num_gradient) + '_4k_' + name + '.dscalar.nii'))

#------------------------------------------------------------------------------
# Save subcortical part
#------------------------------------------------------------------------------

for num_gradient in range(num_components_g):
    # Load the 4mm MNI template
    template_4mm_path = 'template_4mm.nii'
    template_4mm_img = nib.load(path_templates + template_4mm_path)
    template_4mm_data = template_4mm_img.get_fdata()
    template_4mm_data[np.where(template_4mm_data != 0)] = full_gradients[8004:, num_gradient]
    modified_img = nib.Nifti1Image(template_4mm_data, template_4mm_img.affine, template_4mm_img.header)
    nib.save(modified_img, path_results + 'gradient_' + str(num_gradient) + '_' + name +'_subcortical_low_res.nii.gz')

#------------------------------------------------------------------------------
# Create concatinated data - for bisualization and comparisons in the next codes
#------------------------------------------------------------------------------

for num_gradient in range(num_components_g):
    command = path_wb_command + 'wb_command -volume-resample ' + \
          path_results + 'gradient_' + str(num_gradient) + '_' + name + '_subcortical_low_res.nii.gz' + \
          ' ' + path_fsLR + 'MNI152_T2_2mm.nii.gz ' + \
          'CUBIC '+ path_results + 'gradient_' + str(num_gradient) + \
          '_' + name +'_subcortical_low_res_2mm.nii -affine ' + path_fsLR + 'affine.txt'
    os.system(command)

    command = path_wb_command + 'wb_command -cifti-create-dense-from-template ' + \
        path_templates + 'cortex_subcortex.dscalar.nii ' + \
         path_results + 'gradient_' + str(num_gradient) + '_' + name + '.dscalar.nii ' + \
        ' -volume-all ' +  path_results + 'gradient_' + str(num_gradient) + \
        '_' + name + '_subcortical_low_res_2mm.nii -cifti ' + \
         path_results + 'gradient_' + str(num_gradient) + '_4k_' + name + '.dscalar.nii'
    os.system(command)

#------------------------------------------------------------------------------
# END