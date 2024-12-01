from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import quadratic_assignment
from smt import sampling_methods

from library_prep_plate_prep.plates import Plate


@dataclass
class PlateAssignmentProblem:
    D: np.ndarray
    F: np.ndarray


class Sampler(ABC):
    
    def seed_control_wells(self, nt: int, num_columns: int, num_rows: int, **kwargs):
        xlimits = np.array(
            [[0., num_columns], [0., num_rows]]
        )
        return self(nt, xlimits, **kwargs)

    @abstractmethod
    def __call__(self, nt: int, xlimits: np.ndarray, **kwargs):
        pass

class LHSampler(Sampler):
    
    def __init__(self, criterion: str):
        self.criterion = criterion

    @staticmethod
    def _snap_to_grid(arr):
        # NOTE: Constrain to grid by rounding
        return arr.round(0)
    
    def __call__(self, nt: int, xlimits: np.ndarray, random_state=None):
        sampler = sampling_methods.LHS(
            xlimits=xlimits,
            criterion=self.criterion,
            random_state=random_state,
        )
        x_y = sampler(int(nt))
        x_y = self._snap_to_grid(x_y)
        
        return x_y

class LHSampler_CenterMaximin(LHSampler):

    def __init__(self):
        super().__init__('cm')


class Solver(ABC):
    
    @abstractmethod
    def __call__(self, prob: PlateAssignmentProblem):
        pass
    

class QAP_2opt(Solver):
    def __call__(self, prob, seed=None, partial_match=None, partial_guess=None, **kwargs):        
        res = quadratic_assignment(
            prob.D,
            prob.F,
            method='2opt',
            options = {
                'rng': seed,
                'partial_match': partial_match,
                'partial_guess': partial_guess,
            }
        )
        return res['col_ind']