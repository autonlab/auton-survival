### Utility functions to find the maximum treatment effect phenotype and mean differential survival
import sys
sys.path.append('../auton_survival/')

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sksurv.metrics import concordance_index_ipcw, brier_score, integrated_brier_score
from sksurv.util import Surv
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def plot_synthetic_data(outcomes, features, interventions):
    import matplotlib.pyplot as plt
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"
    fs = 48 # Font size
    s = 65 # Size of the marker
    lim = 2.25

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(16,8), sharey=True)
    color_maps = {
        0: LinearSegmentedColormap.from_list("z1", colors=['black', 'C0']),
        1: LinearSegmentedColormap.from_list("z1", colors=['black', 'r']),
        2: LinearSegmentedColormap.from_list("z1", colors=['black', 'r'])
    }
    for cmap in color_maps: color_maps[cmap].set_gamma(0.4)

    # Data
    X1, X2, X3, X4 = features.X1.to_numpy(), features.X2.to_numpy(), features.X3.to_numpy(), features.X4.to_numpy()

    # First sub-plot X1 vs X2
    for z in set(outcomes.Z):
        mask = (outcomes.Z == z)
        sns.kdeplot(ax=ax1, x=X1[mask], y=X2[mask],
                    fill=False,  levels=10, thresh=0.3,
                    cmap=color_maps[z])

    ax1.tick_params(axis="both", labelsize=21)
    ax1.set_xlabel( r'$\mathcal{X}_1 \longrightarrow$', fontsize=fs)
    ax1.set_ylabel( r'$\mathcal{X}_2 \longrightarrow$', fontsize=fs)
    ax1.text(-2,0.5, s=r'$\mathcal{Z}_1$', color='C0', fontsize=fs, 
             bbox=dict(lw=2, boxstyle="round", ec='C0', fc=(.95, .95, .95)))
    ax1.text(1,1.75, s=r'$\mathcal{Z}_2$', color='C2', fontsize=fs, 
             bbox=dict(lw=2, boxstyle="round", ec='C2', fc=(.95, .95, .95)))
    ax1.text(1,-1.75, s=r'$\mathcal{Z}_3$', color='C3', fontsize=fs, 
             bbox=dict(lw=2, boxstyle="round", ec='C3', fc=(.95, .95, .95)))
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)

    # Second sub-plot X1 vs X2
    R = (np.abs(X3) + np.abs(X4))<=2
    ax2.scatter(X3[R], X4[R], s=s, c='white', marker='X',ec='C0')
    ax2.scatter(X3[~R], X4[~R], s=s, c='white', marker='o', ec='C3')

    grid = np.meshgrid([2, 1, 0, -1, -2], [2, 1, 0, -1, -2])
    ax1.scatter(grid[0].ravel(), grid[1].ravel(), color='grey', marker='+', zorder=-500, s=50)

    ax2.set_xlabel(r'$\mathcal{X}_3 \longrightarrow$', fontsize=fs)
    ax2.set_ylabel(r'$\mathcal{X}_4 \longrightarrow$', fontsize=fs)
    ax2.text(-1.25,.25, s=r'$\phi_1$', color='C0', fontsize=fs, 
             bbox=dict( lw=2, boxstyle="round", ec='C0', fc=(.95, .95, .95)))
    ax2.text(1,-1.75, s=r'$\phi_2$', color='C3', fontsize=fs, 
             bbox=dict( lw=2, boxstyle="round", ec='C3', fc=(.95, .95, .95)))
    ax2.tick_params(axis="both", labelsize=21)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    
    plt.show()

def factual_evaluate(train_data, test_data, horizons, predictions):
    """
    Function to evaluate the Concordance indices and Integrated brier score
    """
    y_train = Surv.from_arrays(train_data[2], train_data[1])
    y_test = Surv.from_arrays(test_data[2], test_data[1])

    y_train_t_max = np.max([row[1] for row in y_train])
    y_test_t_vidx = (np.array([row[1] for row in y_test])<y_train_t_max)

    y_test = y_test[y_test_t_vidx]
    predictions = predictions[y_test_t_vidx] 

    concordance_indexes = []

    for i, horizon in enumerate(horizons):
        score = concordance_index_ipcw(y_train, y_test, 1-predictions[:, i], tau=horizon)
        concordance_indexes.append(float(score[0]))

    ibs = integrated_brier_score(y_train, y_test, predictions, times=horizons) 

    return concordance_indexes + [ibs]

def find_max_treatment_effect_phenotype(g, zeta_probs, factual_outcomes):
    """
    Find the group with the maximum treatement effect phenotype
    """
    mean_differential_survival = np.zeros(zeta_probs.shape[1]) # Area under treatment phenotype group
    outcomes_train, interventions_train = factual_outcomes 

    # Assign each individual to their treatment phenotype group
    for gr in range(g): # For each treatment phenotype group
        # Probability of belonging the the g^th treatment phenotype
        zeta_probs_g = zeta_probs[:, gr] 
        # Consider only those individuals who are in the top 75 percentiles in this phenotype
        z_mask = zeta_probs_g>np.quantile(zeta_probs_g, 0.75) 

        mean_differential_survival[gr] = find_mean_differential_survival(
            outcomes_train.loc[z_mask], interventions_train.loc[z_mask])

    return np.nanargmax(mean_differential_survival)

def find_mean_differential_survival(outcomes, interventions):
    """
    Given outcomes and interventions, find the maximum restricted mean survival time
    """
    from lifelines import KaplanMeierFitter

    treated_km = KaplanMeierFitter().fit(outcomes['uncensored time treated'].values, np.ones(len(outcomes)).astype(bool))
    control_km = KaplanMeierFitter().fit(outcomes['uncensored time control'].values, np.ones(len(outcomes)).astype(bool))

    unique_times = treated_km.survival_function_.index.values.tolist() + control_km.survival_function_.index.values.tolist()  
    unique_times = np.unique(unique_times)

    treated_km = treated_km.predict(unique_times, interpolate=True)
    control_km = control_km.predict(unique_times, interpolate=True)

    mean_differential_survival = np.trapz(y=(treated_km.values - control_km.values),
                                        x=unique_times)

    return mean_differential_survival

def plot_phenotypes_roc(outcomes, zeta_probs):
    from matplotlib import pyplot as plt
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

    zeta = outcomes['Zeta']

    y_true = zeta == 0

    fpr, tpr, thresholds = roc_curve(y_true, zeta_probs)
    auc = roc_auc_score(y_true, zeta_probs) 

    plt.figure(figsize=(5,5))

    plt.plot(fpr, tpr, label="AUC: "+str(round(auc, 3)), c='darkblue')
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), ls='--', color='k')

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24, loc='upper left') 

    plt.xlabel('FPR', fontsize=36)
    plt.ylabel('TPR', fontsize=36)
    plt.xscale('log')
    plt.show()