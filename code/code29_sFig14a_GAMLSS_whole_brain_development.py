"""

GAMLSS-RS iteration 1: Global Deviance = 2287.042 
GAMLSS-RS iteration 2: Global Deviance = 2274.182 
GAMLSS-RS iteration 3: Global Deviance = 2269.69 
GAMLSS-RS iteration 4: Global Deviance = 2268.035 
GAMLSS-RS iteration 5: Global Deviance = 2267.417 
GAMLSS-RS iteration 6: Global Deviance = 2267.187 
GAMLSS-RS iteration 7: Global Deviance = 2267.095 
GAMLSS-RS iteration 8: Global Deviance = 2267.055 
GAMLSS-RS iteration 9: Global Deviance = 2267.041 
GAMLSS-RS iteration 10: Global Deviance = 2267.036 
GAMLSS-RS iteration 11: Global Deviance = 2267.034 
GAMLSS-RS iteration 12: Global Deviance = 2267.032 
GAMLSS-RS iteration 13: Global Deviance = 2267.032 
0.3847856121155294

--- Male GAMLSS Model Summary ---
Family: ['GG' 'generalised Gamma Lopatatsidis-Green']
Parameters: ['mu', 'sigma', 'nu']
Method: RS, Converged: True, Iterations: 13
Observations: 290, df.fit: 11, df.residual: 279
AIC: 2289.03, BIC (SBC): 2329.40, Deviance: 2267.03
Link functions - mu: log, sigma: log
******************************************************************
Family:  c("GG", "generalised Gamma Lopatatsidis-Green") 

Call:  gamlss(formula = y ~ fp(x), sigma.formula = ~fp(x),  
    nu.formula = ~1, family = GG(), data = df) 

Fitting method: RS() 

------------------------------------------------------------------
Mu link function:  log
Mu Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  4.39875    0.01084   405.7   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

------------------------------------------------------------------
Sigma link function:  log
Sigma Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept) -1.94062    0.04373  -44.37   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

------------------------------------------------------------------
Nu link function:  identity 
Nu Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)    4.705      0.727   6.472 4.33e-10 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

------------------------------------------------------------------
NOTE: Additive smoothing terms exist in the formulas: 
 i) Std. Error for smoothers are for the linear effect only. 
ii) Std. Error for the linear terms maybe are not accurate. 
------------------------------------------------------------------
No. of observations in the fit:  290 
Degrees of Freedom for the fit:  11
      Residual Deg. of Freedom:  279 
                      at cycle:  13 
 
Global Deviance:     2267.032 
            AIC:     2289.032 
            SBC:     2329.4 
******************************************************************
None
GAMLSS-RS iteration 1: Global Deviance = 2681.326 
GAMLSS-RS iteration 2: Global Deviance = 2671.341 
GAMLSS-RS iteration 3: Global Deviance = 2669.454 
GAMLSS-RS iteration 4: Global Deviance = 2669.032 
GAMLSS-RS iteration 5: Global Deviance = 2668.935 
GAMLSS-RS iteration 6: Global Deviance = 2668.91 
GAMLSS-RS iteration 7: Global Deviance = 2668.905 
GAMLSS-RS iteration 8: Global Deviance = 2668.904 
GAMLSS-RS iteration 9: Global Deviance = 2668.903 
0.20560183121633124

--- Female GAMLSS Model Summary ---
Family: ['GG' 'generalised Gamma Lopatatsidis-Green']
Parameters: ['mu', 'sigma', 'nu']
Method: RS, Converged: True, Iterations: 9
Observations: 337, df.fit: 11, df.residual: 326
AIC: 2690.90, BIC (SBC): 2732.92, Deviance: 2668.90
Link functions - mu: log, sigma: log
******************************************************************
Family:  c("GG", "generalised Gamma Lopatatsidis-Green") 

Call:  gamlss(formula = y ~ fp(x), sigma.formula = ~fp(x),  
    nu.formula = ~1, family = GG(), data = df) 

Fitting method: RS() 

------------------------------------------------------------------
Mu link function:  log
Mu Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  4.48783    0.01001   448.4   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

------------------------------------------------------------------
Sigma link function:  log
Sigma Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)  -1.9616     0.0404  -48.56   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

------------------------------------------------------------------
Nu link function:  identity 
Nu Coefficients:
            Estimate Std. Error t value Pr(>|t|)    
(Intercept)   4.1806     0.7068   5.915 8.37e-09 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

------------------------------------------------------------------
NOTE: Additive smoothing terms exist in the formulas: 
 i) Std. Error for smoothers are for the linear effect only. 
ii) Std. Error for the linear terms maybe are not accurate. 
------------------------------------------------------------------
No. of observations in the fit:  337 
Degrees of Freedom for the fit:  11
      Residual Deg. of Freedom:  326 
                      at cycle:  9 
 
Global Deviance:     2668.903 
            AIC:     2690.903 
            SBC:     2732.924 
******************************************************************

Note: Related to Fig.14a.

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

df = pd.read_csv(path_info_sub + 'Dev_clean_data_info.csv')
num_subjects = len(df)
age = np.array(df['interview_age'])/12
sex = df.sex

#------------------------------------------------------------------------------
# Load data (vertex-wise)
#------------------------------------------------------------------------------

data_vertexwise = np.load(path_results + 'Dev_perfusion_all_vertex.npy')

#------------------------------------------------------------------------------
# Stratify data by sex
#------------------------------------------------------------------------------

data_men = data_vertexwise[:, sex == 'M']
data_women = data_vertexwise[:, sex == 'F']

age_men = age[sex == 'M']
age_women = age[sex == 'F']

# Load your data
data_vertexwise = np.load(path_results + 'Dev_perfusion_all_vertex.npy')
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
plt.savefig(path_figures + 'fig0_dev_only.svg', dpi = 300)
plt.show()

#------------------------------------------------------------------------------
# END
