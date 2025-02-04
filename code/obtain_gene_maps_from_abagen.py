"""

Extract the transcriptomics from the Abagen toolbox.
Genes with differential stability less than 0.1 are not included in the analysis.

The aim is to use the gene data later to run gene enrichment and also to construct the
layer IV-specific signature.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import abagen
import globals
import numpy as np
import pandas as pd
from scipy.io import savemat
from globals import path_genes
from nilearn.datasets import fetch_atlas_schaefer_2018

#------------------------------------------------------------------------------
# Extract gene name and gene spatial patterns from abagen toolbox
#------------------------------------------------------------------------------

nnodes = globals.nnodes_Schaefer
parcellation = fetch_atlas_schaefer_2018(n_rois = nnodes)['maps']

expression = abagen.get_expression_data(parcellation,
                                        lr_mirror     = 'bidirectional',
                                        missing       = 'interpolate',
                                        return_donors = True)

expression_st, ds = abagen.correct.keep_stable_genes(list(expression.values()),
                                                      threshold        = 0.1,
                                                      percentile       = False,
                                                      return_stability = True)

expression_st = pd.concat(expression_st).groupby('label').mean()
columns_name = np.array(expression_st.columns)
data_to_save = {'names': columns_name}
savemat(path_genes + 'names_genes_schaefer_400_filtered.mat', data_to_save)

data_to_save = {'gene_coexpression': np.array(expression_st)}
savemat(path_genes + 'gene_coexpression_schaefer_400_filtered.mat', data_to_save)

#------------------------------------------------------------------------------
# END