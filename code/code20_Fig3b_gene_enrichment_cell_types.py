"""

Cell-type enrichment analysis

Note: Related to Fig.3b.

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

name = 'GCEA_pearsonr_v1_0_GO-PsychEncode-cellTypesUMI-discrete'
cell = pd.read_csv(path_resultsgene + name + '.csv')
pval = 0.05

# Sort results
#cell_filtered = cell.sort_values(by = ['pValZ'], ascending = False).reset_index(drop = True)
cell_filtered = cell.sort_values(by = ['cScorePheno'], ascending = False).reset_index(drop = True)

# Set up figure
fig, ax = plt.subplots(figsize = (5, 10))

# Plot as dashed lines with circles at the end
for i, (score, label) in enumerate(zip(cell_filtered['cScorePheno'],
                                       cell_filtered['cLabel'])):
    y_position = i  # Position on the y-axis

    # Dashed line
    ax.plot([0, score], [y_position, y_position], 'k--',
            linewidth = 1)

    # Thicker line for significant entries
    if cell_filtered['pValPermCorr'].iloc[i] < pval:
        ax.plot([0, score], [y_position, y_position], 'k--',
                linewidth=2.5)

    # Circle at the end of the line
    ax.plot(score, y_position, 'ko',
            markersize = 5)  # 'ko' is a black circle marker

    # Add label with significance marking
    display_label = label + '*' if cell_filtered['pValPermCorr'].iloc[i] < pval else label
    weight = 'bold' if cell_filtered['pValPermCorr'].iloc[i] < pval else 'normal'
    ax.text(score,
            y_position,
            display_label,
            ha = 'left',
            va = 'center',
            size =  12,
            weight = weight)

# Configure plot appearance
ax.axvline(0, color='black', linewidth = 1)
ax.set_xlabel('cScorePheno', fontsize = 12)
ax.set_yticks(range(len(cell_filtered)))
ax.set_yticklabels([])  # Remove y-axis labels
ax.tick_params(left=False)
sns.despine(left = True, bottom = True)

# Save the plot
plt.savefig(path_figures + 'cell_types_enrichment.svg',
            bbox_inches = 'tight',
            dpi = 300)

#------------------------------------------------------------------------------
# END