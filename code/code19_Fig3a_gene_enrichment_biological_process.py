"""

Biological processes enrichment analysis

Note: Related to Fig.3a.

"""
#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from globals import path_resultsgene, path_figures

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Load data and start processing
#------------------------------------------------------------------------------

name = 'GCEA_pearsonr_v1_30_GO-biologicalProcessDirect-discrete'
cell = pd.read_csv(path_resultsgene + name + '.csv')

# Sort data
cell_filtered = cell.sort_values(by = ['pValPermCorr'], ascending = True).reset_index(drop = True)
cell_filtered = cell_filtered.sort_values(by = ['cScorePheno'], ascending = False).reset_index(drop = True).head(10)

pval = 0.05

# Set up figure
fig, ax = plt.subplots(figsize = (5, 8))

# Plot as dashed lines with circles at the end
for i, (score, label) in enumerate(zip(cell_filtered['cScorePheno'],
                                       cell_filtered['cDesc'])):
    y_position = i  # Position on the y-axis
    
    # Dashed line
    ax.plot([0, score], [y_position, y_position], 'k--',
            linewidth = 1)
    
    # Thicker line for significant entries
    if cell_filtered['pValPermCorr'].iloc[i] < pval:
        ax.plot([0, score], [y_position, y_position], 'k--',
                linewidth = 2.5)

    # Circle at the end of the line
    ax.plot(score, y_position, 'ko', markersize = 5)  # 'ko' is a black circle marker
    
    # Add label with significance marking
    display_label = label + '*' if cell_filtered['pValPermCorr'].iloc[i] < pval else label
    weight = 'bold' if cell_filtered['pValPermCorr'].iloc[i] < pval else 'normal'
    ax.text(score,
            y_position,
            display_label,
            ha = 'left',
            va = 'center',
            size = 12,
            weight = weight)

# Configure plot appearance
ax.axvline(0, color = 'black', linewidth = 1)
ax.set_yticks(range(len(cell_filtered)))
ax.set_yticklabels([])  # Remove y-axis labels
ax.tick_params(left = False)
sns.despine(left = True, bottom = True)

# Save the plot
plt.savefig(path_figures + 'biological_terms_enrichment.svg',
 bbox_inches = 'tight',
 dpi = 300)

#------------------------------------------------------------------------------
# END