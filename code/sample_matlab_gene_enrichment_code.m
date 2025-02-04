clc
close all;
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
startup;

addpath('/Results/');
datasets                 = '/Results/';

abagen_schaefer400       = load('gene_coexpression_filtered.mat') % from abagen
expression_matrix        = abagen_schaefer400.gene_coexpression;

names_abagen_schaefer400 = load('names_genes_filtered.mat') % from abagen
gene_symbols             = names_abagen_schaefer400.names;
save('aba_mat.mat','expression_matrix','gene_symbols');

name                     = 'GO-biologicalProcessDirect-discrete' %PsychEncode-cellTypesUMI-discrete' % can be any of the two
opt.GCEA.dataset         = name;
opt.analysis_name        = 'GO-biologicalProcessDirect-discrete'; % adjust accordingly
spin_path                = 'perfusion_pc0_vasa_cortical_matlab.mat'; % PC spatially autocorrelated nulls
Y_spin                   = load(spin_path).spin_res;
nullMaps                 = Y_spin';
save('spins.mat', 'nullMaps');
brain_map_path          = 'perfusion_pc0_map_cortical_matlab.mat'; # PC map
brain_map               = load(brain_map_path).perfusion;
opt.phenotype_data      = brain_map;
opt.phenotype_nulls     ='spins.mat';
opt.n_nulls             = 50000 # Number of spins
opt.dir_result          = ['/Results/' ...
    ''];
opt.atlas               = 'Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz'; % parcellation
opt.aba_mat             = 'aba_mat.mat';
opt.GCEA.size_filter    = [30, 200]; % gene range
opt.GCEA.correlation_method = 'Pearson'; % type of correlation
opt.GCEA.p_tail             = 'right';
cTable = ABAnnotate(opt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% END
