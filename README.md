# Cerebral blood perfusion across biological systems and the human lifespan
Authors: Asa Farahani, Zhen-Qi Liu, Eric G. Ceballos, Justine Y. Hansen, Karl Wennberg, Yashar Zieghami, Mahsa Dadar, Claudine J. Gauthier, Alain Dagher, Bratislav Misic.

This repository contains the code used to generate the results in the "Cerebral blood perfusion across biological systems and the human lifespan" maniuscript.

## Data Confidentiality Notice
The Arterial Spin Labeling (ASD) dataset used in this study is provided by the HCP Lifespan studies, and cannot be publicly released by us. For more details and to request access to the dataset, please visit the [HCP Lifespan](https://www.humanconnectome.org/lifespan-studies) website.

## Repository Structure
### Codes
This folder contains all scripts used in the project. The filenames indicate their corresponding figures/analysis in the manuscript.

#### Utility Scripts
- `globals.py` - Defines the paths to data directories and some constants used throughout the project.
- `functions.py` - Contains functions utilized across various scripts in the project.
- `obtain_gene_maps_from_abagen.py` - Retrieves gene data using the [abagen](https://github.com/rmarkello/abagen) package.
- `resample_ATT/CBF_HCP_lifespan.py` - Resamples data before gradient analysis.
- `combine_data_HCPA\HCPD.py` - Aggregates blood perfusion, arterial transit time (ATT), and cortical thickness data across subjects.
- `parcellate_neurosynth_maps.py` - Parcellates [Neurosynth](https://neurosynth.org) maps.
- `parcellate_disease_maps.py' - Parcellates [disease maps](https://pmc.ncbi.nlm.nih.gov/articles/PMC5740544/).
- `create_FC_HCPA\HCPD.py` - Constructs functional connectomes per subject from resting-state functional MRI data.
  
#### Requirements
- `Requirements.txt` - Contains information about the Python version, system configurations, and the environment on which the project was developed.

### Data
This folder contains data essential for running the analysis, including: [Neurosynth](https://neurosynth.org) maps, [Disease maps](https://pmc.ncbi.nlm.nih.gov/articles/PMC5740544/), [Parcellation files](https://github.com/yetianmed/subcortex/tree/master/Group-Parcellation), [Vessel territories map](https://www.nature.com/articles/s41597-022-01923-0/), etc.
Note: Some directories are empty due to data-sharing restrictions (e.g., subject_information; FCs, behavioral_measures_vitals_raw).

### Results
#### Results_organized
Contains results that can be shared (group averaged files)

01 - Mean cerebral blood perfusion maps (for male & female groups)
02 - Blood perfsuion principal component maps
03 - Arterial transit time (ATT) principal component maps
04 - GLM results applied on HCP-development and HCP-aging combined (for blood perfusion)
05 - Blood perfsuion principal component maps (after regressing out age & sex effects)
06 - First principal component map of functional connectome strength
07 - GLM results applied on HCP-development (age & sex effects on blood perfusion) 
08 - GLM results applied on HCP-development (age & sex effects on cortical thickness) 
09 - GLM results applied on HCP-aging (age & sex effects on blood perfusion) 
10 - Spearman correlation between age & cerebral blood perfusion maps (male & female groups)
11 - PLS loading maps
12 - Gradients of blood perfusion covariance matrix (arterial territories)

#### Results_gene
Outputs from gene enrichment analysis using [Abanotate](https://github.com/LeonDLotter/ABAnnotate).

### Supplementrary_items
Contains suppelemntrary items of the maniuscript. 

## Contact Information
For questions, please email: [asa.borzabadifarahani@mail.mcgill.ca](mailto:asa.borzabadifarahani@mail.mcgill.ca).
