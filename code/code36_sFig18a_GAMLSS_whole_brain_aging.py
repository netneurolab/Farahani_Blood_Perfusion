"""

Non-linear trajectory of blood perfusion in HCP-Aging dataset.

Note: Related to Fig.S18a.

"""

#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------

import warnings
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import matplotlib.pyplot as plt
from IPython import get_ipython
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from globals import path_figures, path_info_sub, path_results

#------------------------------------------------------------------------------
# Configuration
#------------------------------------------------------------------------------

warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')

#------------------------------------------------------------------------------
# Load subject information
#------------------------------------------------------------------------------

df = pd.read_csv(path_info_sub + 'clean_data_info.csv')
num_subjects = len(df)
age = np.array(df['interview_age'])/12
sex = df.sex

#------------------------------------------------------------------------------
# Load data (vertex-wise)
#------------------------------------------------------------------------------

data_vertexwise = np.load(path_results + 'perfusion_all_vertex.npy')

#------------------------------------------------------------------------------
# Stratify data by sex
#------------------------------------------------------------------------------

data_men = data_vertexwise[:, sex == 'M']
data_women = data_vertexwise[:, sex == 'F']

age_men = age[sex == 'M']
age_women = age[sex == 'F']

# Load your data
data_vertexwise = np.load(path_results + 'perfusion_all_vertex.npy')
sex = np.array(sex)  # ensure it's a numpy array
age = np.array(age)

#------------------------------------------------------------------------------
# Fit GAMLSS for each sex group
#------------------------------------------------------------------------------

# Activate R <-> pandas conversion
pandas2ri.activate()

# Load R packages
gamlss = importr('gamlss')
base = importr('base')
stats = importr('stats')

results = {}

cmap = plt.get_cmap('coolwarm')
colors = cmap(np.linspace(0, 1, 100))
sex_labels = [('Male', 'M', colors[10]), ('Female', 'F', colors[90])]

for label, sex_code, color in sex_labels:
    mask = sex == sex_code
    x = age[mask]
    y = np.mean(data_vertexwise[:, mask], axis=0)

    df_r = pd.DataFrame({'x': x, 'y': y})
    ro.globalenv['df'] = pandas2ri.py2rpy(df_r)

    #ro.r('model <- gamlss(y ~ fp(x), sigma.fo = ~ fp(x), data = df, family = GG())')
    ro.r('library(gamlss)')
    ro.r('model <- gamlss(y ~ fp(x), sigma.fo = ~ fp(x), nu.fo = ~1, data = df, family = GG())')
    mu_fitted = np.array(ro.r('fitted(model, what = "mu")'))

    mu_formula_str = " ".join(ro.r('as.character(formula(model, what="mu"))'))
    sigma_formula_str = " ".join(ro.r('as.character(formula(model, what="sigma"))'))

    # Compute R² and adjusted R²
    ss_res = np.sum((y - mu_fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared_mu = 1 - ss_res / ss_tot
    n = int(ro.r('model$N')[0])
    k = int(ro.r('model$df.fit')[0])
    r2_adj_mu = 1 - ((1 - r_squared_mu) * (n - 1)) / (n - k - 1)
    print(r2_adj_mu)

    # Extract core model info
    model_family = str(ro.r('model$family'))
    model_params = list(ro.r('model$parameters'))
    gdev = float(ro.r('model$G.deviance')[0])
    df_fit = int(ro.r('model$df.fit')[0])
    df_resid = int(ro.r('model$df.residual')[0])
    aic = float(ro.r('AIC(model)')[0])
    bic = float(ro.r('GAIC(model, k=log(length(df$y)))')[0])
    converged = bool(ro.r('model$converged')[0])
    iterations = int(ro.r('model$iter')[0])
    method = str(ro.r('model$method')[0])
    n_obs = int(ro.r('model$N')[0])
    link_mu = str(ro.r('model$mu.link')[0])
    link_sigma = str(ro.r('model$sigma.link')[0])

    print(f"\n--- {label} GAMLSS Model Summary ---")
    print(f"Family: {model_family}")
    print(f"Parameters: {model_params}")
    print(f"Method: {method}, Converged: {converged}, Iterations: {iterations}")
    print(f"Observations: {n_obs}, df.fit: {df_fit}, df.residual: {df_resid}")
    print(f"AIC: {aic:.2f}, BIC (SBC): {bic:.2f}, Deviance: {gdev:.2f}")
    print(f"Link functions - mu: {link_mu}, sigma: {link_sigma}")
    print(ro.r('summary(model)'))
    
    # Store for plotting
    results[label] = {'x': x, 'y': y, 'mu': mu_fitted, 'color': color}
    
#------------------------------------------------------------------------------
# Plot results
#------------------------------------------------------------------------------

plt.figure(figsize=(6, 5))

for label in results:
    x = results[label]['x']
    y = results[label]['y']
    mu = results[label]['mu']
    color = results[label]['color']

    plt.scatter(x, y, alpha=0.6, label=f'{label} Observed', color=color)
    plt.plot(np.sort(x), mu[np.argsort(x)], color=color, linewidth=2, label=f'{label} Fitted μ')

plt.xlabel("Age (years)")
plt.ylabel("Mean Perfusion")
plt.title("Sex-stratified GAMLSS Fit (Mean Perfusion vs Age)")
plt.legend()
plt.tight_layout()
plt.savefig(path_figures + 'fig0_aging_only.svg', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END
