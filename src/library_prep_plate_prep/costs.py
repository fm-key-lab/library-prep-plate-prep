from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

__all__ = ['SamplesCostFn', 'SameFamily', 'CovarSimilarity', 'PlateCostFn', 'SqEuclidean']


class CostFn(ABC):
    """Base class for all costs."""

    @abstractmethod
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute cost between :math:`x` and :math:`y`.

        Args:
          x: Array.
          y: Array.

        Returns:
          The cost.
        """

    def all_pairs(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute matrix of all pairwise costs.

        Args:
          x: Array of shape ``[n, ...]``.
          y: Array of shape ``[m, ...]``.

        Returns:
          Array of shape ``[n, m]`` of cost evaluations.
        """
        return np.array([[self(x_, y_) for y_ in y] for x_ in x])

    def h(self, z: np.ndarray):
        """Transformation-invariant function.

        Args:
        z: Array of shape ``[d,]``.

        Returns:
        The cost.
        """


class SamplesCostFn(CostFn):
    """Base class for costs over sequencing samples."""

    def all_pairs(self, sample_data: pd.DataFrame) -> np.ndarray:
        return np.array([[self(x_, y_) for y_ in sample_data.itertuples()] for x_ in sample_data.itertuples()])

    @abstractmethod
    def __call__(self, x: namedtuple, y: namedtuple) -> float:
        pass


class SameFamily(SamplesCostFn):
    """Same family."""

    def __call__(self, x: namedtuple, y: namedtuple):
        return ~int(x.family == y.family)


class CovarSimilarity(SamplesCostFn):
    """Covariate similarity."""

    _costs = np.array([1, 2, 4, 10])
    _rules = ['family', 'family_&_timepoint', 'family_&_donor', 'family_&_donor_&_timepoint']
    
    @property
    def costs(self):
        return -1 * self._costs

    @costs.setter
    def costs(self, vals):
        self._costs = np.array(vals)

    def __call__(self, x: namedtuple, y: namedtuple) -> float:

        def one_is_control_or_blank():
            return (x.family == -1) or (y.family == -1)

        def same_donor():
            return x.donor == y.donor

        def same_timepoint():
            return x.timepoint == y.timepoint

        def same_family():
            return x.family == y.family

        if one_is_control_or_blank():
            return 0

        elif not same_family():
            return 0

        else:
            if same_donor() and same_timepoint():
                return self.costs[3]

            elif same_donor():
                return self.costs[2]

            elif same_timepoint():
                return self.costs[1]

            else:
                return self.costs[0]

    @classmethod
    def from_rules(cls, rules):
        cfn = cls()
        cfn.costs = [rules[r] for r in cfn._rules]
        return cfn

    def as_rules(self):
        return {r: float(c) for r, c in zip(self._rules, -self.costs)}


class PlateCostFn(CostFn):
    """Base class for costs over one or more plates."""

    def all_pairs(self, x_y: np.ndarray) -> np.ndarray:
        return np.array(self(x_y))

    @abstractmethod
    def __call__(self, x, y) -> float:
        pass


class SqEuclidean(PlateCostFn):
    """Squared Euclidean distance."""

    def __call__(self, z):
        return distance_matrix(z, z)