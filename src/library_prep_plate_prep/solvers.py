from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import quadratic_assignment


@dataclass
class PlateAssignmentProblem:
    D: np.ndarray
    F: np.ndarray


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