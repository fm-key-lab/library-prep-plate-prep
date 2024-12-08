from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import quadratic_assignment
from smt import sampling_methods

from library_prep_plate_prep import costs, problems


class Solver(ABC):
    def __init__(self, **kwargs):
        ...

    def __call__(self, prob: problems.QuadraticProblem, **kwargs):
        return self.output_from_state(self.run(prob, **kwargs), **kwargs)

    @abstractmethod
    def run(self, _geom_xx_cost, _geom_yy_cost, **kwargs):
        """"""

    @abstractmethod
    def output_from_state(self, state, **kwargs):
        """"""


class LHSampler(Solver):
    """smt implementation of LHS (center-maximin criterion)."""

    def run(self, prob, nt, random_state=None, **kwargs):
        """Perform LHS."""        
        inds = []
        for i, plate in enumerate(prob._geom_plate._plates):
            xlims = np.array([[1., getattr(plate, rc)] for rc in ['columns', 'rows']])
            sampler = sampling_methods.LHS(xlimits=xlims, criterion='cm', random_state=random_state)
            ind = self._snap_to_grid(sampler(int(nt[i])))
            inds.append(np.pad(ind, ((0,0), (1,0)), constant_values=i))
        
        return {
            'seed': np.concatenate(inds),
            'plate_ind': prob._geom_plate.p_x_y,
            'ctrl_ind': np.argwhere(prob._geom_samples.data.index == 'control').flatten(),
        }

    def output_from_state(self, state, **kwargs):
        # TODO: Parse to partial_match format
        plate_seed = np.argmax((state['plate_ind'] == state['seed'][:, None]).sum(-1), 1)
        return np.c_[plate_seed, state['ctrl_ind']]

    @staticmethod
    def _snap_to_grid(arr):
        """Constrain to grid by rounding."""
        return arr.round(0).astype(int)
    

class QAP_2opt(Solver):
    """scipy implementation of 2-opt method for solving QAPs."""

    def run(self, prob, rng=None, partial_match=None, **kwargs):
        """Wrapper for `scipy.optimize.quadratic_assignment(method='2opt')`."""
        opts = {'rng': rng, 'partial_match': partial_match}
        return quadratic_assignment(
            prob._geom_plate.cost_matrix, prob._geom_samples.cost_matrix, method='2opt', options=opts
        )

    def output_from_state(self, state, **kwargs):
        return state['col_ind']