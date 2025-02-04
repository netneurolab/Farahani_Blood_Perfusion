"""

Define constants and paths needed for the project

"""
#------------------------------------------------------------------------------
# Numeric constants
#------------------------------------------------------------------------------

num_vertices_voxels = 91282 # Num CIFTI vertices/voxels (cortex-subcortex)
num_cort_vertices_noMW = 59412 # Num CIFTI vertices (cortex-only)
num_subcort_voxels = 31870 # Num voxels in subcortical mask (91282 - 59412)

num_cort_vertices_withMW = 64984 # Num vertices in right and left GIFTI files
num_vertices_gifti = 32492 # Num vertices in a single hemisphere GIFTI file (right or left)

nnodes_Schaefer = 400 # Num parcels in schaefer-400 parcellation
nnodes_Glasser = 360 # Num parcels in Glasser multi-modal parcellation
nnodes_Glasser_half = 180 # Num parcels in Glasser multi-modal parcellation (one hemisphere only)
nnodes_Schaefer_S4 = 454 # Num parcels in Schaefer-400 and Tian S4 atlas (cortex and subcortex)

nnodes_S4 = 54 # Num parcels in Tian S4 atlas (subcortex)
nnodes_half_subcortex_S4 = 27 # Num parcels in Tian S4 atlas - one hemisphere (subcortex)

#------------------------------------------------------------------------------
# Paths needed for the analysis
#------------------------------------------------------------------------------

# Raw HCP data directories
path_mri       = '/media/afarahani/Expansion/HCP_DATA_fromServer/'
path_ASL_aging = '/media/afarahani/Expansion/HCP_DATA_fromServer/HCP_ASL/'
path_mri_dev   = '/media/afarahani/Expansion/DATA_ASL_raw_development/'

# Path for HCP-wb_command
path_wb_command = '/home/afarahani/Downloads/workbench/bin_linux64/'

# Paths where results, data, and figures are stored for this project
my_pc = '/home/afarahani/Desktop/blood_annotation/Farahani_blood_perfusion/'
path_results     = my_pc + 'results/'
path_resultsgene = my_pc + 'results_gene/'
path_data        = my_pc + 'data/'
path_figures     = my_pc + 'figures/'

# Path for Schaefer-Tian atlas
path_atlas         = path_data + 'schaefer_tian/Cortex-Subcortex/' # atlas as a CIFTI file
path_atlasV        = path_data + 'schaefer_tian/Cortex-Subcortex/MNIvolumetric/' # atlas as a MNI-volumetric file
parcel_names_S4    = path_data + 'schaefer_tian/' + 'NAME_S4.txt' # Name of parcels in Tian-S4 parcellation
path_coord         = path_data + 'schaefer_400/' # Schaefer-400 parcel coordinates - needed for spin generation
path_yeo           = path_data + 'yeo/' # Yeo7Networks atlas - Schaefer-400
path_atlas_disease = path_data + 'atlas_disease/' # Schaefer-400 in TPM space (original volume space for disease maps)

# Path to files needed for CIFTI read/write (or e.g. resampling)
path_medialwall = path_data + 'medialwall/' # Mask for CIFTI medial wall vertices
path_templates  = path_data + 'templates/' # Path to template CIFTI files
path_fsLR       = path_data + 'fsLR_transform_files/'
path_surface    = path_data + 'GA_surface_files/' # Path for HCP-surface files (.surf.gii)

# Additional data paths
path_disease           = path_data + 'disease/' # Pathologically confirmed disease maps parcellated according to Schaefer-400
path_disease_raw       = path_data + 'disease_raw/' # Raw disease maps
path_info_sub          = path_data + 'subject_information/' # CSV files - participant-related information
path_measures          = path_data + 'behavioral_measures_vitals_raw/' # CSV files - participant-related behavioral/blood test results
path_FC                = path_data + 'FCs/' # Functional connectome of individual subjects
path_glasser           = path_data + 'glasser_parcellation/' # Glasser multi-modal atlas
path_receptors         = path_data + 'receptor_hansen/' # Parcellated receptor maps - from Justine Hansen
path_neurosynth        = path_data + 'neurosynth/' # Parcellated neurosynth maps
path_neurosynth_raw    = path_data + 'neurosynth_raw/' # Raw neurosynth maps
path_vessel            = path_data + 'vessel_territories/' # Arterial territory maps in CIFTI format
path_vessel_raw        = path_data + 'vessel_territories_raw/Atlas_182_MNI152/' # Volumetric arterial territory maps
path_genes             = path_data + 'gene_expression/' # Genes coming from abagen toolbox - originally coming from AHBA

#------------------------------------------------------------------------------
# END