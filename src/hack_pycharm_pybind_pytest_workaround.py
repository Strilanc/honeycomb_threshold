import stim

_problem_repr = stim.Circuit.__repr__
stim.Circuit.__repr__ = lambda e: _problem_repr(e)
