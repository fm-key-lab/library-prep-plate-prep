from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import quadratic_assignment
from smt import sampling_methods

from library_prep_plate_prep.plates import Plate


class Sampler(ABC):
    """Base class for all plate geometry samplers."""
    
    def seed_control_wells(self, nt: int, num_columns: int, num_rows: int, **kwargs):
        """Arrange sample on plate."""
        xlimits = np.array(
            [[0., num_columns], [0., num_rows]]
        )
        return self(nt, xlimits, **kwargs)

    @abstractmethod
    def __call__(self, nt: int, xlimits: np.ndarray, **kwargs):
        pass

class LHSampler(Sampler):
    """smt implementation of LHS."""

    def __init__(self, criterion: str):
        self.criterion = criterion

    @staticmethod
    def _snap_to_grid(arr):
        """Constrain to grid by rounding."""
        return arr.round(0)
    
    def __call__(self, nt: int, xlimits: np.ndarray, random_state=None):
        """Perform LHS."""
        sampler = sampling_methods.LHS(
            xlimits=xlimits,
            criterion=self.criterion,
            random_state=random_state,
        )
        x_y = sampler(int(nt))
        x_y = self._snap_to_grid(x_y)
        
        return x_y

class LHSampler_CenterMaximin(LHSampler):
    """smt implementation of LHS (center-maximin criterion)."""

    def __init__(self):
        super().__init__('cm')


class Solver(ABC):
    """Base class for all solvers."""
    
    @abstractmethod
    def __call__(self, D, F):
        pass
    

class QAP_2opt(Solver):
    """scipy implementation of 2-opt method for solving QAPs."""
    
    def __call__(self, D, F, seed=None, partial_match=None, partial_guess=None, **kwargs):
        """Wrapper for scipy.optimize.quadratic_assignment(method='2opt')."""
        res = quadratic_assignment(
            D,
            F,
            method='2opt',
            options = {
                'rng': seed,
                'partial_match': partial_match,
                'partial_guess': partial_guess,
            }
        )
        
        return res['col_ind']