"""

rev 1 - Dominance analysis

# include gene expression pattern for layers
    # p-value: 9.990009990009990001e-04
    # np.sum(dominance, axis = 1) --> 0.64122314

# include Receptors
    # p-value: 9.990009990009990001e-04
    # np.sum(dominance, axis = 1) --> 0.59081914

NOTE: Related to Fig.S9.

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import os
import globals
import scipy.io
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
import matplotlib.pyplot as plt
from netneurotools import stats
from functions import vasa_null_Schaefer
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import squareform, pdist
from statsmodels.stats.multitest import multipletests
from nilearn.datasets import fetch_atlas_schaefer_2018
from functions import convert_cifti_to_parcellated_SchaeferTian
from globals import path_results, path_genes, path_figures, path_receptors, path_coord

#------------------------------------------------------------------------------
# Needed functions
#------------------------------------------------------------------------------

def get_reg_r_sq(X, y):
    
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    yhat = lin_reg.predict(X)
    SS_Residual = sum((y - yhat) ** 2)
    SS_Total = sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (float(SS_Residual)) / SS_Total
    adjusted_r_squared = 1 - (1 - r_squared) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    return adjusted_r_squared

def cv_slr_distance_dependent(X, y, coords, train_pct = .75, metric = 'rsq'):
    
    '''
    cross validates linear regression model using distance-dependent method.
    X = n x p matrix of input variables
    y = n x 1 matrix of output variable
    coords = n x 3 coordinates of each observation
    train_pct (between 0 and 1), percent of observations in training set
    metric = {'rsq', 'corr'}
    '''
    P = squareform(pdist(coords, metric = "euclidean"))
    train_metric = []
    test_metric = []

    for i in range(len(y)):
        distances = P[i, :]  # for every node
        idx = np.argsort(distances)

        train_idx = idx[:int(np.floor(train_pct * len(coords)))]
        test_idx = idx[int(np.floor(train_pct * len(coords))):]

        mdl = LinearRegression()
        mdl.fit(X[train_idx, :], y[train_idx])
        if metric == 'rsq':
            # get r^2 of train set
            train_metric.append(get_reg_r_sq(X[train_idx, :], y[train_idx]))

        elif metric == 'corr':
            rho, _ = pearsonr(mdl.predict(X[train_idx, :]), y[train_idx])
            train_metric.append(rho)

        yhat = mdl.predict(X[test_idx, :])
        if metric == 'rsq':
            
            # get r^2 of test set
            SS_Residual = sum((y[test_idx] - yhat) ** 2)
            SS_Total = sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
            r_squared = 1 - (float(SS_Residual)) / SS_Total
            adjusted_r_squared = 1-(1-r_squared)*((len(y[test_idx]) - 1) /
                                                  (len(y[test_idx]) -
                                                   X.shape[1] - 1))
            test_metric.append(adjusted_r_squared)

        elif metric == 'corr':
            rho, _ = pearsonr(yhat, y[test_idx])
            test_metric.append(rho)
    return train_metric, test_metric

def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.nanmean(null))
                    > abs(emp - np.nanmean(null)))) / (len(null) + 1)

def get_reg_r_pval(X, y, spins, nspins):
    
    emp = get_reg_r_sq(X, y)
    null = np.zeros((nspins, ))
    for s in range(nspins):
        null[s] = get_reg_r_sq(X[spins[:, s], :], y)
    return (1 + sum(null > emp))/(nspins + 1)

#------------------------------------------------------------------------------
# Load perfsuion data - the main map of interst
#------------------------------------------------------------------------------

scores_data = np.load(path_results + 'perfusion_PCscore.npy')[:,0]
perfusion = convert_cifti_to_parcellated_SchaeferTian(scores_data.reshape(1, globals.num_vertices_voxels),
                                                      'cortex',
                                                      'S1',
                                                      path_results,
                                                      'test')

#------------------------------------------------------------------------------
# Load neuropeptides
#------------------------------------------------------------------------------

nnodes = 400
parc_file_mni = fetch_atlas_schaefer_2018(n_rois = nnodes)['maps']
cortex = np.arange(nnodes)

expression_st = scipy.io.loadmat(path_genes + 'gene_coexpression_schaefer_400_filtered.mat')['gene_coexpression']
name_genes = scipy.io.loadmat(path_genes + 'names_genes_schaefer_400_filtered.mat')['names']

columns_name = name_genes
n_genes = len(expression_st.T)

expression_st = np.array(expression_st)
df_corr_features = pd.DataFrame({
    'genes': columns_name.flatten(),
})

index = np.where((columns_name == 'EDNRA') |
                 (columns_name == 'EDNRB') |
                 (columns_name == 'VIPR1') |
                 (columns_name == 'VIPR2')
                 )[1]
num_genes = len(index)
neuropeptides_data = expression_st[:,index] # 400 by 4
neuropeptides_names = ['EDNRA', 'EDNRB', 'VIPR1', 'VIPR2']

#------------------------------------------------------------------------------
# Load neurotransmitters
#------------------------------------------------------------------------------

df = pd.read_csv(path_receptors + 'receptors_ctx.csv')
receptor_cols = [col for col in df.columns if col != "labels"]
receptor_names = receptor_cols
receptor_data = df[receptor_cols].values  # 400 by 7
num_receptors = len(receptor_names)

#------------------------------------------------------------------------------
# Load layer maps
#------------------------------------------------------------------------------

layer_names = ['III','IV', 'V']
layer_data = np.zeros((nnodes, len(layer_names)))
c = 0
for layer in layer_names:
    path_gene_maps_abagen = os.path.join(path_genes, 'names_genes_schaefer_400_filtered.mat')
    gene_maps_abagen_raw = loadmat(path_gene_maps_abagen)['names']
    gene_maps_abagen = gene_maps_abagen_raw.ravel()
    if isinstance(gene_maps_abagen[0], bytes):
        gene_maps_abagen = [g.decode('utf-8') for g in gene_maps_abagen]
    else:
        gene_maps_abagen = gene_maps_abagen.tolist()
    print(f"Number of genes in Abagen data: {len(gene_maps_abagen)}")

    path_data_abagen = os.path.join(path_genes, 'gene_coexpression_schaefer_400_filtered.mat')
    data_abagen = loadmat(path_data_abagen)['gene_coexpression']
    print("data_abagen shape:", data_abagen.shape)  
    if layer == 'III':
        cleaned_gene_list = {'C1QL2', 'C20orf103', 'CARTPT', 'DISC1', 'GLRA3',
                            'GSG1L', 'IGSF11', 'INPP4B', 'MFGE8', 'PVRL3', 
                            'RASGRF2','SV2C','WFS1'}
    elif layer == 'IV':
        cleaned_gene_list = {'COL6A1', 'CUX2', 'TRMT9B', 'GRIK4', 'RORB'}
    elif layer == 'V':
        cleaned_gene_list = {'ADRA2A', 'AKR1C3', 'ANXA1', 'B3GALT2', 'CDH24',
                             'CTGF', 'ETV1', 'FAM3C', 'FOXP2', 'HTR2C',
                             'KCNK2', 'NPY2R', 'NR4A2', 'NTNG2', 'OPRK1',
                             'PCDH17', 'PCDH20', 'PCP4', 'PDE1A', 'RPRM',
                             'RXFP1', 'SNTB1', 'SYT10', 'SYT6', 'TLE4',
                             'TOX', 'TRIB2', 'VAT1L'}
    cleaned_gene_list = list(cleaned_gene_list) # Ensure it's a list for indexing
    common_genes = [g for g in cleaned_gene_list if g in gene_maps_abagen]
    missing_genes = [g for g in cleaned_gene_list if g not in gene_maps_abagen]
    idx_common = [gene_maps_abagen.index(g) for g in common_genes]
    layer_datas = data_abagen[:, idx_common]
    zscored_layer_data = zscore(layer_datas, axis = 0)
    layer_data[:,c] = np.mean(zscored_layer_data, axis = 1) #400 by 3
    c = c + 1

#------------------------------------------------------------------------------
# Combine the results to one another: neurotransmitter + neuropeptides + layer information
#------------------------------------------------------------------------------

#data = np.concatenate((layer_data, receptor_data, neuropeptides_data), axis = 1) #(400, 14)
#data_names = np.concatenate((layer_names, receptor_names, neuropeptides_names)) #14

data = np.concatenate((receptor_data, neuropeptides_data), axis = 1) #(400, 11)
data_names = np.concatenate((receptor_names, neuropeptides_names)) #11

#------------------------------------------------------------------------------
# Do spins and load correlation
#------------------------------------------------------------------------------

nspins = 1000
spins = vasa_null_Schaefer(nspins)
coords = np.genfromtxt(path_coord + 'Schaefer_400.txt')
coords = coords[:, -3:]

#------------------------------------------------------------------------------
#                              Dominance Analysis 
#------------------------------------------------------------------------------

model_metrics = dict([])
train_metric = np.zeros([nnodes, 1])
test_metric = np.zeros(train_metric.shape)

model_metrics, _ = stats.get_dominance_stats(zscore(data),
                                             zscore(perfusion))
# cross validate the model
[train_metric , test_metric] = cv_slr_distance_dependent(zscore(data), 
                                                         zscore(perfusion),
                                                         coords,
                                                         0.75,
                                                         metric = 'rsq')
# get p-value of model
model_pval = get_reg_r_pval(zscore(data),
                            zscore(perfusion), 
                            spins, 
                            nspins)

model_pval = multipletests(model_pval, 
                           method='fdr_bh')[1]

dominance = np.zeros((1, len(data_names)))
dominance[0, :] = model_metrics["total_dominance"]

plt.ion()
plt.figure()
plt.bar(np.arange(1), 
        np.sum(dominance, axis = 1),
        tick_label = 'perfusion score map')
plt.xticks(rotation = 'vertical')
plt.tight_layout()
plt.savefig(path_figures + 'rev1_dominance_molecular_2.svg', dmi = 300)
dominance[np.where(model_pval >= 0.05)[0], :] = 0

plt.ion()
plt.figure()
sns.heatmap(dominance / np.sum(dominance, axis = 1)[:, None],
            cmap = 'coolwarm',
            vmin= -0.15, vmax = 0.15,
            xticklabels = data_names,linewidth = 1.5)
plt.savefig(path_figures + 'rev1_dominance_molecular_1.svg', dmi = 300)
plt.tight_layout()

# plot cross validation results
fig, (ax1, ax2) = plt.subplots(2, 1)
sns.violinplot(data = train_metric, ax = ax1)
sns.violinplot(data = test_metric, ax = ax2)
ax1.set(ylabel = 'train set correlation', ylim = (-1, 1))
ax2.set_xticklabels('perfusion score map', rotation = 90)
ax2.set(ylabel = 'test set correlation', ylim = (-1, 1))
plt.tight_layout()
plt.savefig(path_figures + 'rev1_dominance_molecular.svg', dmi = 300)

#------------------------------------------------------------------------------
# END
