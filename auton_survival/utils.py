import inspect

def _get_method_kwargs(method, kwargs):

  assert isinstance(kwargs, dict)

  params = inspect.signature(method).parameters.items()
  params = set([param[0] for param in params]) - set(['self'])

  method_params = params&set(kwargs.keys())
  method_kwargs = {k: kwargs[k] for k in method_params}

  return method_kwargs
