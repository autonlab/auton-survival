import inspect
import pandas as pd

def _get_method_kwargs(method, kwargs):

  assert isinstance(kwargs, dict)

  params = inspect.signature(method).parameters.items()
  params = set([param[0] for param in params]) - set(['self'])

  method_params = params&set(kwargs.keys())
  method_kwargs = {k: kwargs[k] for k in method_params}

  return method_kwargs

def _dataframe_to_array(data):
  if isinstance(data, (pd.Series, pd.DataFrame)):
    return data.to_numpy()
  else:
    return data

# TaR: Code alternative to lifelines.utils.qth_percentile
from scipy.optimize import fsolve
def interp_x(y, x, thres):
  if len(y[y<thres])==0:
    x1, y1 = 0, y[0]
  else:
    x1, y1 = x[y<thres][-1], y[y<thres][-1]
    x2, y2 = x[y>thres][0], y[y>thres][0]
    root = fsolve(func, x0=x1, args=(thres, x1, y1, x2, y2))[0]
  return root
def func(x, y, x1, y1, x2, y2):
  return y1 + (x-x1)*((y2-y1)/(x2-x1)) - y
