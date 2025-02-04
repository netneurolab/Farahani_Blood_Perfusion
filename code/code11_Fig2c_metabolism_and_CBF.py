"""
# Assess the correspondence of the first PC of CBF with PET data (Glucose and O₂ energy metabolism maps)

cmr02
rho    = 0.5222192638703992
r      = 0.6248796179328594
p_spin = 0.000999

cmruglu
rho    = 0.6949762811017569
r      = 0.7380904965195442
p_spin = 0.000999

Note: Related to Fig.2c.

"""
#-------------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------------

import globals
import scipy.io
import warnings
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from neuromaps import datasets, transforms
from scipy.stats import spearmanr, pearsonr
from functions import save_as_dscalar_and_npy
from numpy.polynomial.polynomial import Polynomial
from functions import vasa_null_Schaefer, pval_cal
from netneurotools.datasets import fetch_schaefer2018
from neuromaps.images import load_data, dlabel_to_gifti
from globals import path_figures, path_medialwall, path_results
from functions import convert_cifti_to_parcellated_SchaeferTian
from functions import save_parcellated_data_in_SchaeferTian_forVis

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

# Use precomputed null distributions (select 1).
flag_null = 0

#------------------------------------------------------------------------------
# Parcellated metabolic maps
#------------------------------------------------------------------------------

mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask'].astype(np.float32)

# Fetch the cortical atlas from the Schaefer parcellation
schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
atlas = load_data(dlabel_to_gifti(schaefer))
yeo_cortical = atlas[(mask_medial_wall.flatten()) == 1]

# List of descriptors (metabolic markers)
descriptors = ['cmr02', 'cmruglu']

# Dictionary to store the left and right hemisphere data
hemisphere_data = {}

# Loop through each descriptor, fetch, transform, and load data
for desc in descriptors:
    # Fetch the annotation
    annotation = datasets.fetch_annotation(source = 'raichle', desc = desc)
    # Transform the annotation to 32k surface density
    annotation_32k = transforms.fslr_to_fslr(annotation,
                                             target_density = '32k',
                                             hemi = {'L', 'R'},
                                             method = 'linear')
    left_data = load_data(annotation_32k[0])
    right_data = load_data(annotation_32k[1])
    hemisphere_data[f'{desc}_l'] = left_data
    hemisphere_data[f'{desc}_r'] = right_data
    data = np.concatenate((left_data, right_data))
    data_59k = data[(mask_medial_wall == 1).flatten()]
    save_as_dscalar_and_npy(data_59k, 'cortex', path_results, desc)
    data_parcellated = np.zeros((globals.nnodes_Schaefer, 1))
    for n in range(1, globals.nnodes_Schaefer + 1):
        data_parcellated[n - 1,:] = np.nanmean(data_59k[yeo_cortical == n])
    hemisphere_data[f'{desc}_parcellated'] = data_parcellated

#------------------------------------------------------------------------------
# Create scatter plot and visualize the correlation values
#------------------------------------------------------------------------------

PC_data  = np.load(path_results + 'perfusion_PCscore.npy')[:, 0]

# Also save it as a brain map
PC_data_parcellated = convert_cifti_to_parcellated_SchaeferTian(PC_data.reshape(1, globals.num_vertices_voxels),
                                          'cortex',
                                          'X',
                                          path_results,
                                          'PC0')

save_parcellated_data_in_SchaeferTian_forVis(PC_data_parcellated,
                                             'cortex',
                                             'S1',
                                             path_results,
                                             'PC_data_parcellated')
for desc in descriptors:
    print(desc)
    print (spearmanr(hemisphere_data[f'{desc}_parcellated'].flatten(),
                     PC_data_parcellated.flatten()))
    print (pearsonr(hemisphere_data[f'{desc}_parcellated'].flatten(),
                    PC_data_parcellated.flatten()))

#------------------------------------------------------------------------------
# vasa nulls
#------------------------------------------------------------------------------

numspins = 1000
if flag_null == 0:
    nulls_index_vasa = vasa_null_Schaefer(numspins)
    np.save(path_results + 'spin_metabolism.npy', nulls_index_vasa)
else: # Use precomputed null distributions (select 1).
   nulls_index_vasa =  np.load(path_results + 'spin_metabolism.npy')

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

num_bio = len(descriptors)
p_value_vasa  = np.zeros((num_bio, 1))
n = 0
for desc in descriptors:
    print({desc})
    bio_data_column = hemisphere_data[f'{desc}_parcellated'].flatten()
    r, generated_null = corr_spin(bio_data_column.T,
                                 PC_data_parcellated,
                                 nulls_index_vasa,
                                 numspins)
    p_value_vasa[n, 0] = pval_cal(r, generated_null, numspins)
    n = n + 1

#------------------------------------------------------------------------------
# Plot significant results - at cortical level
#------------------------------------------------------------------------------

def plot_significant_scatter(x, y, xlabel, ylabel, title, file_name, name_bio_plot):
    plt.figure(figsize = (5, 5))
    plt.scatter(x, y,
                color = 'gray',
                s = 15)
    p = Polynomial.fit(x, y, 1)
    plt.plot(*p.linspace(),
             color = 'black',
             linewidth = 1)
    plt.ylim(-56, 66)
    plt.title(str(name_bio_plot))
    plt.yticks(np.arange(-56, 66, 10))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_name,
                format = 'svg')
    plt.show()
j = 0
for desc in descriptors:
    if p_value_vasa[j, 0] < 0.05:
        print(desc)
        print(pearsonr(zscore(hemisphere_data[f'{desc}_parcellated'].flatten()),
                                 PC_data_parcellated.flatten()))
        plot_significant_scatter(zscore(hemisphere_data[f'{desc}_parcellated']).flatten(),
                                 PC_data_parcellated.flatten(),
                                 str(desc),
                                 'Mean Perfusion',
                                  str(desc) + ' vs. Mean Perfusion',
                                 path_figures + str(desc) + '_vs_perfusion_cortical.svg',
                                 desc)
    j = j + 1
#------------------------------------------------------------------------------
# END