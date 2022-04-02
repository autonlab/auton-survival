import numpy as np
import pandas as pd

from lifelines import KaplanMeierFitter, NelsonAalenFitter

from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts


def plot_kaplanmeier(outcomes, groups=None, plot_counts=False, **kwargs):

  """Plot a Kaplan-Meier Survival Estimator stratified by groups.

  Parameters
  ----------
  outcomes: pandas.DataFrame
    a pandas dataframe containing the survival outcomes. The index of the
    dataframe should be the same as the index of the features dataframe.
    Should contain a column named 'time' that contains the survival time and
    a column named 'event' that contains the censoring status.
    \( \delta_i = 1 \) if the event is observed.
  groups: pandas.Series
    a pandas series containing the groups to stratify the Kaplan-Meier
    estimates by.
  plot_counts: bool
    if True, plot the number of at risk and censored individuals in each group.

  """

  if groups is None:
    groups = np.array([1]*len(outcomes))

  curves = {}

  from matplotlib import pyplot as plt

  ax = plt.subplot(111)

  for group in sorted(set(groups)):
    if pd.isna(group): continue

    curves[group] = KaplanMeierFitter().fit(outcomes[groups==group]['time'],
                                            outcomes[groups==group]['event'])
    ax = curves[group].plot(label=group, ax=ax, **kwargs)

  if plot_counts:
    add_at_risk_counts(iter([curves[group] for group in curves]), ax=ax)

  return ax


def plot_nelsonaalen(outcomes, groups=None, **kwargs):

  """Plot a Nelson-Aalen Survival Estimator stratified by groups.

  Parameters
  ----------
  outcomes: pandas.DataFrame
    a pandas dataframe containing the survival outcomes. The index of the
    dataframe should be the same as the index of the features dataframe.
    Should contain a column named 'time' that contains the survival time and
    a column named 'event' that contains the censoring status.
    \( \delta_i = 1 \) if the event is observed.
  groups: pandas.Series
    a pandas series containing the groups to stratify the Kaplan-Meier
    estimates by.

  """

  if groups is None:
    groups = np.array([1]*len(outcomes))

  for group in sorted(set(groups)):
    if pd.isna(group): continue

    print('Group:', group)

    NelsonAalenFitter().fit(outcomes[groups==group]['time'],
                            outcomes[groups==group]['event']).plot(label=group,
                                                                   **kwargs)


