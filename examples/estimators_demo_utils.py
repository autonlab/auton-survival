import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

def plot_performance_metrics(results, times):
  """Plot Brier Score, ROC-AUC, and time-dependent concordance index
  for survival model evaluation.

  Parameters
  -----------
  results : dict
      Python dict with key as the evaulation metric
  times : float or list
      A float or list of the times at which to compute
      the survival probability.

  Returns
  -----------
  matplotlib subplots

  """

  colors = ['blue', 'purple', 'orange', 'green']
  gs = gridspec.GridSpec(1, len(results), wspace=0.3)

  for fi, result in enumerate(results.keys()):
    val = results[result]
    x = [str(round(t, 1)) for t in times]
    ax = plt.subplot(gs[0, fi]) # row 0, col 0
    ax.set_xlabel('Time')
    ax.set_ylabel(result)
    ax.set_ylim(0, 1)
    ax.bar(x, val, color=colors[fi])
    plt.xticks(rotation=30)
  plt.show()
