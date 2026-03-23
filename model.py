# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:50:58 2026

@author: utente
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cmdstanpy import CmdStanModel
import arviz as az
import json

with open('piedmont_data.json') as f:
    data = json.load(f)

model = CmdStanModel(stan_file="model1.stan")

# --- Sample from the model ---
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    show_console=True
)

# --- Show summary ---
# s = fit.summary()
# print(s)

idata = az.from_cmdstanpy(fit)

s1 = az.summary(idata, var_names=["alpha","sh","sh_sigma","log_incidence"])
s1.to_csv('figs/m1_summary.csv')

az.plot_trace(idata, var_names=['alpha'])
plt.show()
plt.close()

az.plot_trace(idata, var_names=['sh'])
plt.show()
plt.close()

az.plot_trace(idata, var_names=['sh_sigma'])
plt.show()
plt.close()

az.plot_trace(idata, var_names=['log_incidence'])
plt.show()
plt.close()

su = az.summary(idata, var_names=["sh"])
x = np.arange(su.shape[0])

plt.scatter(x, su['mean'], color='tab:red')
plt.plot([x,x], [su['hdi_3%'], su['hdi_97%']], color='tab:red')
plt.xticks(x, labels=x+1)
plt.grid(alpha=0.3)
plt.title('Subsector coefficients')
plt.xlabel('Subsectors')
plt.savefig('figs/m1_sh.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()

su = az.summary(idata, var_names=["log_incidence"])
x = np.arange(su.shape[0])

plt.scatter(x, np.exp(su['mean']), color='tab:red')
plt.plot([x,x], np.exp([su['hdi_3%'], su['hdi_97%']]), color='tab:red')
plt.xticks(x, labels=x+1)
plt.grid(alpha=0.3)
plt.yscale('log')
plt.title('Incidence by severity')
plt.xlabel('Severity class')
plt.savefig('figs/m1_incid.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()

y_rep = np.array(az.extract(idata, var_names="y_rep"))

sec = np.array(data['sector'])
y_obs = np.array(data['y'])

###############################################################################

mask = (sec == 16)
x = np.arange(8)
bvals = []
for i in x:
    vals = y_rep[mask,i,:].flatten()
    bvals.append(vals)
    plt.hlines(y_obs[mask,i], i-0.45, i+0.45, color='tab:red', lw=1, zorder=5)
plt.boxplot(bvals, positions=x, patch_artist=True, widths=0.7,
            showmeans=False, showfliers=False,
            medianprops={"color": "white", "linewidth": 0.5},
            boxprops={"facecolor": "grey", "edgecolor": "white",
                      "linewidth": 0.5},
            whiskerprops={"color": "grey"},
            capprops={"color": "grey"}, zorder=4)
plt.yscale('log')
plt.grid(alpha=0.3)
plt.title('Metalworking')
plt.xticks(x, labels=x+1)
plt.xlabel('Severity classes')
plt.ylabel('Reported cases (log)')
plt.savefig('figs/m1_metal.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()

###############################################################################

sev = 2
x = np.arange(data['S'])
bvals = []
for s in x:
    mask = (sec == s+1)
    vals = y_rep[mask,sev,:].flatten()
    bvals.append(vals)
    plt.hlines(y_obs[mask,sev], s-0.45, s+0.45, color='tab:red', lw=1, zorder=5)
plt.boxplot(bvals, positions=x, patch_artist=True, widths=0.7,
            showmeans=False, showfliers=False,
            medianprops={"color": "white", "linewidth": 0.5},
            boxprops={"facecolor": "grey", "edgecolor": "white",
                      "linewidth": 0.5},
            whiskerprops={"color": "grey"},
            capprops={"color": "grey"}, zorder=4)
plt.title('Severity class 3')
plt.xticks(x, labels=x+1)
plt.grid(alpha=0.3)
plt.ylabel('Number of injuries')
plt.xlabel('Subsectors')
plt.savefig('figs/m1_sc3.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()

###############################################################################

# MODEL 2

model = CmdStanModel(stan_file="model2.stan")

# --- Sample from the model ---
fit = model.sample(
    data=data,
    chains=4,
    parallel_chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    show_console=True
)

# --- Show summary ---
# s = fit.summary()
# print(s)

idata2 = az.from_cmdstanpy(fit)

s2 = az.summary(idata2, var_names=["alpha","sh","sh_sigma","sh_severity","log_incidence"])
s2.to_csv('figs/m2_summary.csv')

az.plot_trace(idata2, var_names=['alpha'])
plt.show()
plt.close()

az.plot_trace(idata2, var_names=['sh'])
plt.show()
plt.close()

az.plot_trace(idata2, var_names=['sh_sigma'])
plt.show()
plt.close()

az.plot_trace(idata2, var_names=['sh_severity'])
plt.show()
plt.close()

##

stoppe

su1 = az.summary(idata, var_names=["sh"])
su2 = az.summary(idata2, var_names=["sh"])
x = np.arange(su1.shape[0])

plt.scatter(x, su1['mean'], color='tab:red', label='Model 1')
plt.plot([x,x], [su1['hdi_3%'], su1['hdi_97%']], color='tab:red')
plt.scatter(x+0.3, su2['mean'], color='tab:blue', label='Model 2')
plt.plot([x+0.3,x+0.3], [su2['hdi_3%'], su2['hdi_97%']], color='tab:blue')
plt.xticks(x, labels=x+1)
plt.grid(alpha=0.3)
plt.title('Subsector coefficients')
plt.xlabel('Subsectors')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('figs/m2_sh.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()

su = az.summary(idata2, var_names=["sh_severity"])
x = np.arange(su.shape[0])

plt.scatter(x, su['mean'], color='tab:blue')
plt.plot([x,x], [su['hdi_3%'], su['hdi_97%']], color='tab:blue')
plt.xticks(x, labels=x+1)
plt.grid(alpha=0.3)
plt.ylabel('Delta from baseline')
plt.xlabel('Severity class')
plt.savefig('figs/m2_gammas.png', dpi=140, bbox_inches='tight')
plt.show()
plt.close()

y_rep = np.array(az.extract(idata, var_names="y_rep"))

sec = np.array(data['sector'])
y_obs = np.array(data['y'])