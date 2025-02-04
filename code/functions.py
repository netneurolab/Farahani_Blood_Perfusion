"""

Define the needed functions for the project

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import scipy.io
import numpy as np
import nibabel as nib
from netneurotools.stats import gen_spinsamples
from neuromaps.images import load_data, dlabel_to_gifti
from netneurotools.datasets import fetch_schaefer2018, fetch_mmpall
from globals import path_templates, path_atlas, path_medialwall, path_coord
#%%
def save_as_dscalar_and_npy(input_data_voxelwise, type_data, path_save, name_save):
    """
        Save the input data as both a CIFTI dscalar file (.dscalar.nii) for visualization and a NumPy (.npy) file.

        This function takes voxel-wise input data, reshapes it, and saves it in two formats:
        1. A CIFTI dscalar (.dscalar.nii) file, which is useful for visualization in tools like Connectome Workbench.
        2. A NumPy (.npy) file, which stores the data in a raw format for further analysis.

        Args:
            input_data_voxelwise (numpy array): The voxel-wise input data. The data should be a 1D.
            type_data (str): Specifies the type of data being saved. Options are:
                             - 'cortex': Data corresponds to cortical vertices (59k).
                             - 'cortex_subcortex': Data corresponds to both cortical and subcortical vertices/voxels (92k).
            path_save (str): The directory where the output files will be saved.
            name_save (str): The base name for the saved files (without the extension).

        Returns:
            None. The function saves the data in two formats:
                - A .dscalar.nii file for visualization.
                - A .npy file for raw data storage.

        Notes:
            - The function uses predefined templates for 'cortex' and 'cortex_subcortex'.
             (Ensure that the correct template files are available at the specified paths.)
            - The input data will be reshaped to match the expected format of the dscalar file.

        Example Usage:
            - To save cortical data:
                save_as_dscalar_and_npy(cortex_data, 'cortex', path_to_save, 'cortex_output')

            - To save combined cortical and subcortical data:
                save_as_dscalar_and_npy(cortex_subcortex_data, 'cortex_subcortex', path_to_save, 'cortex_subcortex_output')
    """
    # Load the templates for 'cortex' and 'cortex_subcortex'
    template_paths = {
        'cortex': os.path.join(path_templates, 'cortex.dscalar.nii'),
        'cortex_subcortex': os.path.join(path_templates, 'cortex_subcortex.dscalar.nii')}
    templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}

    # Choose the correct template based on the 'type_data' argument
    if type_data == 'cortex':
        template = templates['cortex']
    if type_data == 'cortex_subcortex':
        template = templates['cortex_subcortex']

    # Save the input data as a CIFTI dscalar (.dscalar.nii) file
    new_img = nib.Cifti2Image(input_data_voxelwise.reshape(1, -1),
                              header = template.header,
                              nifti_header = template.nifti_header)
    new_img.to_filename(os.path.join(path_save, name_save + '.dscalar.nii'))

    # Save the input data as a NumPy (.npy) file
    np.save(path_save + name_save + '.npy', input_data_voxelwise)
#%%
def convert_cifti_to_parcellated_SchaeferTian(data, type_data, version, path_save, name_save):
    """
        Convert vertex-wise data to a parcellated format using the Schaefer-Tian atlas (versions S1 or S4).

        This function converts vertex-wise data into parcellated data based on the Schaefer-Tian atlas.
        The function supports parcellation of the cortex (schaefer), subcortex (tian), or both (schaefer-tain), and adjusts the number of parcels depending 
        on the version specified ('S1' or 'S4'). For cortical data, the 'version' argument is ignored.

        Args:
            data (numpy array): The vertex-wise data to be parcellated. The data array should have dimensions of 
                                (number of volumnes (e.g., timepoints), number of vertices/voxels).
                                number of vertices/voxels should be 59k (if only cortex) or 92k to match CIFTI files format.
            type_data (str): Specifies which data should be parcellated.
                             Options are:
                             - 'cortex': Only cortical data will be parcellated.
                             - 'cortex_subcortex': Both cortical and subcortical data will be parcellated.
                             - 'subcortex': Only subcortical data will be parcellated.
                             - 'subcortex_double': Only subcortical data will be parcellated
                                (in this case, homologous parcels in the left and right hemispheres will be considered as one.)
            version (str): Specifies the version of the Schaefer-Tian atlas to use.
                           Options are:
                           - 'S1': Uses 16 subcortical parcels.
                           - 'S4': Uses 54 subcortical parcels.
                           - if cortex is the aim, version can be any string of choice; it will not be incorporated in the naming.
            path_save (str): The directory where the output file will be saved.
            name_save (str): The base name for the saved file.

        Returns:
            numpy array: The parcellated data as a numpy array. The shape depends on the `type_data` and `version` arguments. For example:
                         - If 'cortex', the output will have shape (400, number of timepoints).
                         - If 'cortex_subcortex', the output will have shape (400 + 16/54, number of timepoints).
                         - If 'subcortex', the output will have shape (16/54, number of timepoints).
                         - If 'subcortex_double', the output will have shape ((8/27), number of timepoints).

        Notes:
            - The function uses the Schaefer-Tian atlas for parcellating the data.
            - Ensure that the input data matches the expected shape for the chosen `type_data`.
            - The output is also saved as a NumPy array in a .npy format.

        Example Usage:
            - To convert cortical data using version 'S1':
                parcellated_data = convert_cifti_to_parcellated_SchaeferTian(data, 'cortex', 'S1', path_to_save, 'cortex_data')

            - To convert both cortical and subcortical data using version 'S4':
                parcellated_data = convert_cifti_to_parcellated_SchaeferTian(data, 'cortex_subcortex', 'S4', path_to_save, 'full_data')
        """
    if type_data == 'cortex':
        # For cortical data, ignore version
        version = None  # Set version to None for cortex to ensure it doesn't affect naming
    else:
        # Set the number of nodes for Schaefer-Tian atlas based on the version
        nnodes_Schaefer_version = globals.nnodes_Schaefer_S1 if version == 'S1' else globals.nnodes_Schaefer_S4
        nnodes_version = globals.nnodes_S1 if version == 'S1' else globals.nnodes_S4
        schaefer_tian = nib.load(path_atlas + f'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_{version}.dlabel.nii').get_fdata()
        schaefer_tian_subcortex = schaefer_tian[0, globals.num_cort_vertices_noMW:]

    # Load the medial wall mask and the Schaefer-Tian atlas for the selected version
    mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask'].astype(np.float32)

    # Fetch the cortical atlas from the Schaefer parcellation
    schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
    atlas = load_data(dlabel_to_gifti(schaefer))
    schaefer_tian_cortical = atlas[(mask_medial_wall.flatten()) == 1]

    # Split the data into cortex and subcortex components
    data_cortex = data[:, :globals.num_cort_vertices_noMW]
    if type_data in ['cortex_subcortex', 'subcortex', 'subcortex_double']:
        data_subcortex = data[:, globals.num_cort_vertices_noMW:]

    if type_data == 'cortex':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((globals.nnodes_Schaefer, np.shape(data)[0]))
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_cortex[:, schaefer_tian_cortical == n], axis = 1)

    if type_data == 'cortex_subcortex':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((nnodes_Schaefer_version, np.shape(data)[0]))
        for n in range(1, nnodes_version + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_subcortex[:, schaefer_tian_subcortex == n], axis = 1)
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_parcellated[n - 1 + nnodes_version,:] = np.nanmean(data_cortex[:, schaefer_tian_cortical == n], axis = 1)

    if type_data == 'subcortex':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((nnodes_version, np.shape(data)[0]))
        for n in range(1, nnodes_version + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_subcortex[:, schaefer_tian_subcortex == n], axis = 1)

    if type_data == 'subcortex_double':
        # Initialize the parcellated data array
        data_parcellated = np.zeros((int(nnodes_version/2), np.shape(data)[0]))
        for n in range(1, int(nnodes_version/2) + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_subcortex[:, (schaefer_tian_subcortex == n) | (schaefer_tian_subcortex == n + int(nnodes_version/2))], axis = 1)

    # Save the parcellated data array as a NumPy file and also return it
    file_name_suffix = f"_{type_data}_parcellated" + (f"_{version}" if version else "")
    np.save(os.path.join(path_save, f"{name_save}{file_name_suffix}.npy"), data_parcellated)
    return data_parcellated
#%%
def save_parcellated_data_in_SchaeferTian_forVis(data, type_data, version, path_save, name_save):
    """
    Save a parcel-wise map based on the Schaefer-Tian atlas for visualization in Connectome Workbench.

    This function converts parcel-wise data (array) based on the Schaefer-Tian atlas into a CIFTI dscalar format (.dscalar.nii)
    suitable for visualization in Connectome Workbench. The function supports data for the cortex, subcortex, or both.

    Args:
        data (numpy array): The parcel-wise data to be saved. The expected shape depends on the `type_data` argument:
                            - 'cortex': Data should have shape (400, 1) corresponding to the 400 cortical parcels.
                            - 'cortex_subcortex': Data should have shape (400 + 16/54, 1) (Schaefer-400 + TianS1/TianS4)
                            - 'subcortex': Data should have shape (16/54, 1) (TianS1/TianS4), depending on the version.
        type_data (str): Specifies which type of data is being processed.
                         Options are:
                         - 'cortex': Process and save only cortical data
                         - 'cortex_subcortex': Process and save both cortical and subcortical data
                         - 'subcortex': Process and save only subcortical data
        version (str): Specifies the version of the atlas being used.
                       Options are:
                       - 'S1': Version with 16 subcortical parcels
                       - 'S4': Version with 54 subcortical parcels
        path_save (str): The path where the output file will be saved
        name_save (str): The base name of the saved file

    Returns:
        None. The function saves the parcel-wise data in a .dscalar.nii file format, which can be visualized using 
        Connectome Workbench.

    Notes:
        - The number of parcels in the subcortex depends on the version ('S1' or 'S4'). 'S1' uses 16 parcels, while 'S4' uses 54 parcels.
        - Ensure that the dimensions of data passed as input matches the expected shape for the specified `type_data`.

    Example Usage:
        - To save data for the cortex only:
            save_parcellated_data_in_SchaeferTian_forVis(cortex_data, 'cortex', 'S1', path_to_save, 'cortex_data')

        - To save data for both cortex and subcortex:
            save_parcellated_data_in_SchaeferTian_forVis(full_data, 'cortex_subcortex', 'S4', path_to_save, 'full_data')
    """
    if type_data == 'cortex':
        # For cortical data, ignore version and use the standard Schaefer parcellation
        version = None  # Set version to None for cortex to ensure it doesn't affect naming
    else:
        # Set the number of nodes for Schaefer-Tian atlas based on the version
        nnodes_version = globals.nnodes_S1 if version == 'S1' else globals.nnodes_S4
        yeo_tian = nib.load(path_atlas + f'Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_{version}.dlabel.nii').get_fdata()
        yeo_tian_subcortex = yeo_tian[0, globals.num_cort_vertices_noMW:]


    # Load the medial wall mask and the Schaefer-Tian atlas for the appropriate version
    mask_medial_wall = scipy.io.loadmat(path_medialwall + 'fs_LR_32k_medial_mask.mat')['medial_mask'].astype(np.float32)

    # Fetch the cortical atlas from the Schaefer parcellation
    schaefer = fetch_schaefer2018('fslr32k')[f"{globals.nnodes_Schaefer}Parcels7Networks"]
    atlas = load_data(dlabel_to_gifti(schaefer))
    yeo_tian_cortical = atlas[(mask_medial_wall.flatten()) == 1]

    # Initialize the data array to save
    data_to_save = np.zeros(globals.num_vertices_voxels)

    # Process cortical data
    if type_data == 'cortex':
        data_to_save_cortex = np.zeros(globals.num_cort_vertices_noMW)
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_to_save_cortex[yeo_tian_cortical == n] = data[n - 1]
        data_to_save[:globals.num_cort_vertices_noMW] = data_to_save_cortex

    # Process both cortical and subcortical data
    if type_data == 'cortex_subcortex':
        data_to_save_cortex = np.zeros(globals.num_cort_vertices_noMW)
        data_to_save_subcortex = np.zeros(globals.num_subcort_voxels)
        for n in range(1, nnodes_version + 1):
            data_to_save_subcortex[yeo_tian_subcortex == n] = data[n - 1]
        for n in range(1, globals.nnodes_Schaefer + 1):
            data_to_save_cortex[yeo_tian_cortical == n] = data[n - 1 + nnodes_version]
        data_to_save[globals.num_cort_vertices_noMW:] = data_to_save_subcortex
        data_to_save[:globals.num_cort_vertices_noMW] = data_to_save_cortex

    # Process subcortical data only
    if type_data == 'subcortex':
        data_to_save_subcortex = np.zeros(globals.num_subcort_voxels)
        for n in range(1, nnodes_version + 1):
            data_to_save_subcortex[yeo_tian_subcortex == n] = data[n - 1]
        data_to_save[globals.num_cort_vertices_noMW:] = data_to_save_subcortex
    # Load the appropriate template for saving
    template_paths = {'cortex_subcortex': os.path.join(path_templates, 'cortex_subcortex.dscalar.nii')}
    templates = {key: nib.cifti2.load(path) for key, path in template_paths.items()}
    template = templates['cortex_subcortex']
    # Create and save the new CIFTI image
    new_img = nib.Cifti2Image(data_to_save.reshape(1, -1),
                              header=template.header,
                              nifti_header=template.nifti_header)

    # Save map
    file_name_suffix = f"_{type_data}_parcellated" + (f"_{version}" if version else "")
    new_img.to_filename(os.path.join(path_save, f"{name_save}{file_name_suffix}" +'.dscalar.nii'))
#%%
def convert_cifti_to_parcellated_Glasser(data, hem):
    """
        Convert vertex-wise (cortex-only) data into a parcellated format using the multi-modal Glasser atlas.

        This function converts a vertex-wise data array into a parcellated data array based on the Glasser atlas.
        This function supports conversion for a single hemisphere ('l' for left, 'r' for right) or both hemispheres ('lr')
        When using 'lr_double', the data for both hemispheres will be averaged and projected onto a combined set of parcels.

        Args:
            data (numpy array): The vertex-wise data to be parcellated. The data array should have dimensions of
                                (number of timepoints or features, number of vertices (59k or 92k))

            hem (str): The hemisphere for which the data should be parcellated.
                       Options are:
                       - 'l': Parcellate data for the left hemisphere.
                       - 'r': Parcellate data for the right hemisphere.
                       - 'lr': Parcellate data for the the whole cortex.
                         'lr_double': Parcellate and average data from both hemispheres: n | n+180.

        Returns:
            numpy array: The parcellated data with shape (number of parcels, number of timepoints or features).
                         The number of parcels is 180 for all three cases (lr_double, r, l).
                         The number of parcels is 360 for all three cases (lr).

        Example Usage:
            - To convert left hemisphere data:
                left_parcellated_data = convert_cifti_to_parcellated_Glasser(data_left, 'l')

            - To convert and average both hemispheres:
                combined_parcellated_data = convert_cifti_to_parcellated_Glasser(data_bilateral, 'lr')
        """
    # Fetch the Glasser parcellation
    glasser = fetch_mmpall('fslr32k')
    atlas_glasser = load_data(glasser)
    atlas_glasser_59k = atlas_glasser[atlas_glasser != 0] # len(atlas_glasser_59k):59412

    # Extract the cortical data from the input
    data_cortex = data[:, :globals.num_cort_vertices_noMW]

    # Initialize a variable to store the parcellated data
    data_parcellated = np.zeros((globals.nnodes_Glasser_half, np.shape(data)[0]))

    # Parcellate left hemisphere data
    if hem == 'r':
        for n in range(1, globals.nnodes_Glasser_half + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_cortex[:, atlas_glasser_59k == n], axis = 1)

    # Parcellate right hemisphere data
    if hem == 'l':
        for n in range(globals.nnodes_Glasser_half + 1, globals.nnodes_Glasser + 1):
            data_parcellated[n - globals.nnodes_Glasser_half - 1,:] = np.nanmean(data_cortex[:, atlas_glasser_59k == n], axis = 1)

    # Parcellate and average all cortex
    if hem == 'lr':
        for n in range(1, globals.nnodes_Glasser + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_cortex[:, atlas_glasser_59k == n], axis = 1)

    # Parcellate and average both hemispheres
    if hem == 'lr_double':
        for n in range(1, globals.nnodes_Glasser_half + 1):
            data_parcellated[n - 1,:] = np.nanmean(data_cortex[:, (atlas_glasser_59k == n + globals.nnodes_Glasser_half ) | (atlas_glasser_59k == n)], axis = 1)
    return data_parcellated
#%%
def pval_cal(rho_actual, null_dis, num_spins):
    """
        Calculate p-value - non-parametric method - two-sided

        Args:
            rho_actual (numpy array): Actual correlation coefficient between maps of interest
            null_dis (numpy array): Null distribusion of correlation coefficient
            num_spins (numoy array): Number of spins (e.g., 1000)

        Returns:
            Spinned indices while preserving the spatial autocorrelation
    """
    p_value = (1 + np.count_nonzero(abs((null_dis - np.mean(null_dis))) > abs((rho_actual - np.mean(null_dis))))) / (num_spins + 1)
    return(p_value)
#%%
def vasa_null_Schaefer(nspins):
    """
        Create spatially autocorrelated null maps (for Schaefer-400)

        Args:
            nspins (numpy array): Number of spins (e.g., 1000)

        Returns:
            Spinned indices while preserving the spatial autocorrelation.
    """
    coords = np.genfromtxt(path_coord + 'Schaefer_400.txt')
    coords = coords[:, -3:]
    nnodes = len(coords)
    hemiid = np.zeros((nnodes,))
    hemiid[:int(nnodes/2)] = 1

    spins = gen_spinsamples(coords,
                            hemiid,
                            n_rotate = nspins,
                            seed = 8, #89745571(gene),#552313(all-others),#8(neurosynth*124)
                            method = 'vasa')
    return spins
#%%
def load_nifti(atlas_path):
    """
    Load nifti data
    """
    return nib.load(atlas_path).get_fdata()
#%%
def save_gifti(file, file_name):
    """
        Generate as a .func.gii file - can be used for visualization in Connectome Workbench.
    """
    da = nib.gifti.GiftiDataArray(file, datatype = 'NIFTI_TYPE_FLOAT32')
    img = nib.GiftiImage(darrays = [da])
    nib.save(img, (file_name +'.func.gii'))
#------------------------------------------------------------------------------
# END