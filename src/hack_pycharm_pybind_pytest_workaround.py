"""
There's an annoying bug between pycharm, pytest, and pybind11 where pycharm tries to "helpfully"
describe failed assertions in a better way, which attempts to hash the __repr__ method of the
objects, which goes poorly when they were defined by pybind11.

Importing this file works around the issue by hiding the pybind11 repr methods inside lambdas.
"""

import stim

_problem_repr_circuit = stim.Circuit.__repr__
stim.Circuit.__repr__ = lambda e: _problem_repr_circuit(e)
_problem_repr_dem = stim.DetectorErrorModel.__repr__
stim.DetectorErrorModel.__repr__ = lambda e: _problem_repr_dem(e)
