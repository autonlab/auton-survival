from sksurv.metrics import concordance_index_ipcw, brier_score, cumulative_dynamic_auc
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

def compute_performance_metrics(model, et_tr, x_new, et_new, times):
  """Compute the Brier Score, ROC-AUC, and time-dependent concordance index
  for survival model evaluation.

  Parameters
  -----------
  model : fitted instance of SurvivalModel class
  et_tr : np.array
      A structured array containing the binary event indicator as first field, 
      and time of event or time of censoring as second field.
  x_new : features: pd.DataFrame
      a pandas dataframe with rows corresponding to individual samples
      and columns as covariates.
  et_new : np.array
      A structured array containing the binary event indicator as first field, 
      and time of event or time of censoring as second field.
  times : float or list
      A float or list of the times at which to compute
      the survival probability.
      
  Returns
  -----------
  dict : 
      dict keys include:
      'roc_auc' : Area under the receiver operating characteristic curve
      'cis' : Time-dependent concordance index
      'brier' : Brier Score

  """
    
  # Compute the estimated probability of survival up to the i-th time point
  prob_surv_new = model.predict_survival(x_new, times)
    
  # Compute the estimated probability of experiencing an event.
  prob_risk_new = 1-prob_surv_new
    
  metric_val = dict()
  metric_val['roc_auc'] = []
  metric_val['cis'] = []
  metric_val['brier'] = brier_score(et_tr, et_new, prob_surv_new, times)[1]   
  for i, _ in enumerate(times):
        
    # The risk scores in the j-th column are used to evaluate the j-th time point.
    metric_val['roc_auc'] += [cumulative_dynamic_auc(et_tr, et_new, prob_risk_new[:, i], 
                                                     times[i])[0][0]]
    metric_val['cis'] += [concordance_index_ipcw(et_tr, et_new, prob_risk_new[:, i], 
                                                 times[i])[0]]
        
  return metric_val



def plot_performance_metrics(metrics, times):
  """Plot Brier Score, ROC-AUC, and time-dependent concordance index
  for survival model evaluation.

  Parameters
  -----------
  metrics : python dict
      dict keys include:
      'roc_auc' : Area under the receiver operating characteristic curve
      'cis' : Time-dependent concordance index
      'brier' : Brier Score 
  times : float or list
      A float or list of the times at which to compute
      the survival probability.
      
  Returns
  -----------
  matplotlib subplots

  """
  gs = gridspec.GridSpec(1, 3, wspace=0.5)

  plt.figure(figsize=(14,4))
  ax = plt.subplot(gs[0, 0]) # row 0, col 0 
  ax.set_xlabel('Time', fontsize=12)
  ax.set_ylabel('Brier Score', fontsize=12)
  ax.set_ylim(0, 1)
  plt.plot(times, metrics['brier'], color='black')

  ax = plt.subplot(gs[0, 1]) # row 0, col 1 
  ax.set_xlabel('Time', fontsize=12)
  ax.set_ylabel('Time-dependent Concordance Index', fontsize=12)
  ax.set_ylim(0, 1)
  plt.plot(times, metrics['cis'], color='black')

  ax = plt.subplot(gs[0, 2]) # row 1, span all columns
  ax.set_xlabel('Time', fontsize=12)
  ax.set_ylabel('ROC-AUC', fontsize=12)
  ax.set_ylim(0, 1)
  plt.plot(times, metrics['roc_auc'], color='black')

  plt.show()