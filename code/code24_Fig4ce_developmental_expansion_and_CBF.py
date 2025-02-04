"""

Describe the developmental age effect:

1. unimodal-transmodal differences:

    unimodal-transmodal statistics - parcel-wise:
        t-test:
            T-statistic: [12.96886447]
            P-value: [2.02540175e-30]
        spin-test: n = 1,000
            difference between the two: 0.6191603279230042
            0.000999000999000999

    -- for the vertex-level results:
            T-statistic_vertex: 131.858254100562
            P-value_vertex: 0.0

2. The pattern is enriched for developmental expansion
    n = 1,000
    devexp (this is the reported one in the paper)
    rho =-0.35746179663622896
    r =-0.3783570551501292
    p_spin(r) = 0.000999

    evoexp
    rho =-0.3595878724242027
    r =-0.3251771995649409
    p_spin(r) = 0.00799201

    scalingnih
    rho =-0.2644825905161907
    r = -0.2479536952387854
    p_spin(r) = 0.003996

    scalingpnc
    rho =-0.3578900493128081
    r = 0.3291492752034456
    p_spin(r) = 0.000999

    scalinghcp
    rho =-0.3175626722667017
    r =-0.2753890035240928
    p_spin(r) = 0.000999

Note: Related to Fig.4c,e.

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import warnings
import scipy.io
import numpy as np
import nibabel as nib
from globals import path_yeo
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from neuromaps import datasets, transforms
from scipy.stats import spearmanr, pearsonr
from functions import save_as_dscalar_and_npy
from numpy.polynomial.polynomial import Polynomial
from functions import vasa_null_Schaefer, pval_cal
from netneurotools.datasets import fetch_schaefer2018
from neuromaps.images import load_data, dlabel_to_gifti
from functions import convert_cifti_to_parcellated_SchaeferTian
from globals import path_figures, path_medialwall, path_results, path_surface, path_wb_command

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
flag_null = 0 # Use pre-calculated spins or not

#------------------------------------------------------------------------------
# Load the age effect in development dataset
#------------------------------------------------------------------------------

age_effect = np.load(path_results + 'Dev_betas_age_perfusion.npy')

# Also save it as a brain map
age_effect_parcellated = convert_cifti_to_parcellated_SchaeferTian(
    age_effect.reshape(1, globals.num_vertices_voxels),
    'cortex',
    'X',
    path_results,
    'age_effect_dev')

#------------------------------------------------------------------------------
# Describe based on unimodal-transmodal
#------------------------------------------------------------------------------

# Yeo7Networks
atlas_7Network = np.load(path_yeo + 'Schaefer2018_400Parcels_7Networks.npy')
Yeo_Network_colors = {
    'Network_0': [120/256, 18/256, 134/256],
    'Network_1': [70/256, 130/256, 180/256],
    'Network_2': [0/256, 118/256, 14/256],
    'Network_3': [196/256, 58/256, 250/256],
    'Network_4': [220/256, 248/256, 164/256],
    'Network_5': [230/256, 148/256, 34/256],
    'Network_6': [205/256, 62/256, 78/256],
}
LABEL_ORDER = [
    'visual',
    'somatomotor',
    'dorsal attention',
    'ventral attention',
    'limbic',
    'frontoparietal',
    'default',
]

# Calculate mean age effect per network
yeo_age_effect = np.zeros(7)
for i in range(7):
    indices = np.where(atlas_7Network == i)[0]
    yeo_age_effect[i] = np.mean(age_effect_parcellated[indices])

# Plotting age effect per network
plt.figure()
plt.bar(LABEL_ORDER, yeo_age_effect, color = 'silver')
plt.xlabel('Age')
plt.ylabel('Perfusion')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.savefig(path_figures + 'perfusion_age_effect_development_yeo_networks.svg',
            format = 'svg',
            bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
# Unimodal - Transmodal Analysis
#------------------------------------------------------------------------------

c_unimodal = [67/255, 160/255, 71/255]     # Green for unimodal regions
c_multimodal = [255/255, 140/255, 0/255]   # Orange for transmodal regions
c_diff = [128/255, 128/255, 128/255]

multimodal = np.where((atlas_7Network == 6) | (atlas_7Network == 5))[0]
unimodal = np.where((atlas_7Network == 0) | (atlas_7Network == 1))[0]

# Calculate mean values for unimodal and transmodal regions
yeo_age_effect_unimodal_transmodal = np.zeros(2)
yeo_age_effect_unimodal_transmodal[0] = np.mean(age_effect_parcellated[unimodal])
yeo_age_effect_unimodal_transmodal[1] = np.mean(age_effect_parcellated[multimodal])
diff_age_effect = yeo_age_effect_unimodal_transmodal[0] - yeo_age_effect_unimodal_transmodal[1]

#------------------------------------------------------------------------------
# Unimodal - Transmodal: T-Test - parcel-wise
#------------------------------------------------------------------------------

# Extract perfusion values and perform t-test
unimodal_perfusion = age_effect_parcellated[unimodal]
multimodal_perfusion = age_effect_parcellated[multimodal]
t_stat, p_value = ttest_ind(unimodal_perfusion, multimodal_perfusion)

# Print t-test results
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value}")

# Plot bar chart with individual parcel values
plt.figure(figsize = (3, 6))
plt.bar([0.15, 0.3],
        [np.mean(unimodal_perfusion), np.mean(multimodal_perfusion)],
        color = [c_unimodal, c_multimodal],
        alpha = 0.6,
        width = 0.1)
plt.scatter(np.full(len(unimodal_perfusion), 0.15),
            unimodal_perfusion,
            color = c_unimodal,
            edgecolor = 'k',
            alpha = 0.7,
            label = 'Unimodal Parcels')
plt.scatter(np.full(len(multimodal_perfusion), 0.3),
            multimodal_perfusion,
            color = c_multimodal,
            edgecolor = 'k',
            alpha = 0.7,
            label = 'Transmodal Parcels')
plt.xticks([0.15, 0.3], ['Unimodal', 'Transmodal'])
plt.xlabel('Network Type')
plt.ylabel('Perfusion')
plt.title('Perfusion Age Effect in Unimodal vs Transmodal Regions')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc = 'upper left')
plt.savefig(path_figures + 'perfusion_age_effect_development_parcelwise_unimoal_transmodal.svg',
            format = 'svg',
            bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
# Unimodal - Transmodal: T-Test - voxel-level
#------------------------------------------------------------------------------

# Load the cifti yeo atlas form the GA data provided by HCP
command = path_wb_command + 'wb_command -file-convert -cifti-version-convert ' +\
    path_surface + 'RSN_networks.32k_fs_LR.dlabel.nii 2 ' + path_surface +\
        'version_corrected_RSN_networks.32k_fs_LR.dlabel.nii '
os.system(command)
file_per = os.path.join(path_surface + 'version_corrected_RSN_networks.32k_fs_LR.dlabel.nii')
img_per = nib.cifti2.load(file_per)
RSN_networks = img_per.get_data()
Yeo_networks_vertex = RSN_networks[0, :]

# Load the medial wall mask and the Schaefer-Tian atlas for the selected version
mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask'].astype(np.float32)
yeo_cortical = Yeo_networks_vertex[(mask_medial_wall.flatten()) == 1]

# Save it for visualization 
save_as_dscalar_and_npy(yeo_cortical,
                        'cortex',
                        path_results,
                        'yeo_cortical')
''' 
# Note to myself about vertex_wise_labels
    vision  --> 41
    sensory --> 43
    dorsal att --> 38
    ventral att --> 44
    limbic --> 42
    fronto --> 39
    default --> 40
'''
unimodal_labels_vertex = np.where((yeo_cortical == 41) | (yeo_cortical == 43))[0]
transmodal_labels_vertex = np.where((yeo_cortical == 39) | (yeo_cortical == 40))[0]

# Extract perfusion values and perform t-test
unimodal_perfusion_vertex = age_effect[:globals.num_cort_vertices_noMW][unimodal_labels_vertex]
multimodal_perfusion_vertex = age_effect[:globals.num_cort_vertices_noMW][transmodal_labels_vertex]
t_stat_vertex, p_value_vertex = ttest_ind(unimodal_perfusion_vertex, multimodal_perfusion_vertex)

# Print t-test results
print(f"T-statistic_vertex: {t_stat_vertex}")
print(f"P-value_vertex: {p_value_vertex}")

# Plot bar chart with individual parcel values
plt.figure(figsize = (3, 6))
plt.bar([0.15, 0.3],
        [np.mean(unimodal_perfusion_vertex), np.mean(multimodal_perfusion_vertex)],
        color = [c_unimodal, c_multimodal],
        alpha = 0.6,
        width = 0.1)
plt.scatter(np.full(len(unimodal_perfusion_vertex), 0.15),
            unimodal_perfusion_vertex,
            color = c_unimodal,
            edgecolor = 'k',
            alpha = 0.7,
            label = 'Unimodal vertices')
plt.scatter(np.full(len(multimodal_perfusion_vertex), 0.3),
            multimodal_perfusion_vertex,
            color = c_multimodal,
            edgecolor = 'k',
            alpha = 0.7,
            label = 'Transmodal vertices')
plt.xticks([0.15, 0.3], ['Unimodal', 'Transmodal'])
plt.xlabel('Network Type')
plt.ylabel('Perfusion')
plt.title('Perfusion Age Effect in Unimodal vs Transmodal vertices')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc = 'upper left')
plt.savefig(path_figures + 'perfusion_age_effect_development_vertixwise_unimoal_transmodal.svg',
            format = 'svg',
            bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
# Unimodal - Transmodal: Spin Test - using parcellated data
#------------------------------------------------------------------------------

nspins = 1000
nulls_index_unTrans = vasa_null_Schaefer(nspins)
nulls_age_effect = np.zeros_like(nulls_index_unTrans)

for spin in range(nspins):
    nulls_age_effect[:, spin] = age_effect_parcellated[nulls_index_unTrans[:, spin]].flatten()

diff_null = np.zeros((nspins, 1))
for spin in range(nspins):
    diff_null[spin, 0] = np.mean(nulls_age_effect[:, spin][unimodal]) - np.mean(nulls_age_effect[:, spin][multimodal])
p_value = pval_cal(diff_age_effect, diff_null, nspins)
print(p_value)

# Plot histogram of null distribution
plt.figure(figsize = (8, 6))
plt.hist(diff_null,
         bins = 30,
         color = 'silver',
         edgecolor = 'black',
         alpha = 0.7,
         label = 'Null Distribution')
plt.axvline(diff_age_effect,
            color = 'red',
            linestyle = '--',
            linewidth = 2,
            label = f'Observed Diff ({diff_age_effect:.3f})')
plt.xlabel('Difference in Mean Perfusion (Unimodal - Multimodal)')
plt.ylabel('Frequency')
plt.title('Null Distribution of Differences in Mean Perfusion')
plt.legend()
plt.savefig(path_figures + 'perfusion_age_effect_development_parcelwise_spintest_unimodal_transmodal.svg',
            format = 'svg',
            bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
# Load parcellated developmental maps from neuromaps
#------------------------------------------------------------------------------

mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask'].astype(np.float32)

# Fetch the cortical atlas from the Schaefer parcellation
schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
atlas = load_data(dlabel_to_gifti(schaefer))
yeo_cortical = atlas[(mask_medial_wall.flatten()) == 1]

# List of descriptors to fetch from neuromaps
descriptors = ['devexp', 'evoexp']

# Dictionary to store the left and right hemisphere data
hemisphere_data = {}

# Loop through each descriptor, fetch, transform, and load data
for desc in descriptors:
    # Fetch the annotation
    annotation = datasets.fetch_annotation(source = 'hill2010',
                                           desc = desc)
    # Transform the annotation to 32k surface density
    annotation_32k = transforms.fslr_to_fslr(annotation,
                                             target_density = '32k',
                                             hemi = {'L', 'R'},
                                             method = 'linear')
    left_data = load_data(annotation_32k[0])
    right_data = load_data(annotation_32k[0])
    hemisphere_data[f'{desc}_l'] = left_data
    hemisphere_data[f'{desc}_r'] = right_data
    data = np.concatenate((left_data, right_data))
    data_59k = data[(mask_medial_wall == 1).flatten()]
    save_as_dscalar_and_npy(data_59k, 'cortex', path_results, desc)
    data_parcellated = np.zeros((globals.nnodes_Schaefer, 1))
    for n in range(1, globals.nnodes_Schaefer + 1):
        data_parcellated[n - 1,:] = np.nanmean(data_59k[yeo_cortical == n])
    hemisphere_data[f'{desc}_parcellated'] = data_parcellated

# Another list of descriptors to fetch from neuromaps
descriptors = ['scalingnih', 'scalingpnc', 'scalinghcp']
for desc in descriptors:
    annotation = datasets.fetch_annotation(source = 'reardon2018',
                                           desc = desc,
                                           space = 'civet',
                                           den = '41k')
    annotation_32k = transforms.civet_to_fslr(annotation,
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

descriptors = ['devexp', 'evoexp','scalingnih', 'scalingpnc', 'scalinghcp']

for desc in descriptors:
    print(desc)
    print (spearmanr(hemisphere_data[f'{desc}_parcellated'].flatten(),
                     age_effect_parcellated.flatten()))
    print (pearsonr(hemisphere_data[f'{desc}_parcellated'].flatten(),
                    age_effect_parcellated.flatten()))

#------------------------------------------------------------------------------
# Vvasa nulls to assess the significance of correlations
#------------------------------------------------------------------------------

numspins = 1000
if flag_null == 0:
    nulls_index_vasa = vasa_null_Schaefer(numspins)
    np.save(path_results + 'spin_metabolism.npy', nulls_index_vasa)
else:
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
                                 age_effect_parcellated,
                                 nulls_index_vasa,
                                 numspins)
    p_value_vasa[n, 0] = pval_cal(r, generated_null, numspins)
    n = n + 1

#------------------------------------------------------------------------------
# Plot significant results - at cortical level
#------------------------------------------------------------------------------

atlas_7Network = np.load(path_yeo + "Schaefer2018_400Parcels_7Networks.npy")

c_unimodal = [67/255, 160/255, 71/255]     # Green for unimodal regions
c_multimodal = [255/255, 140/255, 0/255]   # Orange for transmodal regions

multimodal = np.where((atlas_7Network == 6) | (atlas_7Network == 5))[0]
unimodal = np.where((atlas_7Network == 0) | (atlas_7Network == 1))[0]
between = np.where((atlas_7Network == 2) | (atlas_7Network == 3) | (atlas_7Network == 4))[0]

def plot_significant_scatter(x, y, xlabel, ylabel, title, file_name, name_bio_plot):

    # Parcel-wise scatter plot with region coloring
    plt.figure(figsize=(5, 5))
    plt.scatter(x[unimodal],
                y[unimodal],
                color = c_unimodal,
                alpha = 0.7,
                s = 15,
                label = 'Unimodal')
    plt.scatter(x[multimodal],
                y[multimodal],
                color = c_multimodal,
                alpha = 0.7,
                s = 15,
                label = 'Multimodal')
    plt.scatter(x[between],
                y[between],
                color = 'silver',
                alpha = 0.7,
                s = 15,
                label = 'In between')

    p = Polynomial.fit(x, y, 1)
    plt.plot(*p.linspace(),
             color = 'black',
             linewidth = 1)
    plt.title(str(name_bio_plot))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(file_name,
                format = 'svg')
    plt.show()
j = 0
for desc in descriptors:
    if p_value_vasa[j,0] < 0.05:
        print(desc)
        print(pearsonr(hemisphere_data[f'{desc}_parcellated'].flatten(),
                                 age_effect_parcellated.flatten()))
        plot_significant_scatter(hemisphere_data[f'{desc}_parcellated'].flatten(),
                                 age_effect_parcellated.flatten(),
                                 str(desc),
                                 'Mean Perfusion',
                                  str(desc) + ' vs. Mean Perfusion',
                                 path_figures + str(desc) + '_vs_perfusion_cortical.svg',
                                 desc)
    j = j + 1

#------------------------------------------------------------------------------
# END