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
        raise NotImplementedError


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

    _costs = np.arange(6) + 1
    _rules = [
        'species',
        'species_&_family',
        'species_&_donor',
        'species_&_family_&_timepoint',
        'species_&_donor_&_family',
        'species_&_donor_&_family_&_timepoint'
    ]
    def __call__(self, x: namedtuple, y: namedtuple) -> float:
        
        def one_is_control_or_blank():
            return (x.family == -1) or (y.family == -1)

        def same_species():
            return x.species == y.species

        def same_donor():
            return x.donor == y.donor

        def same_family():
            return x.family == y.family

        def same_timepoint():
            return x.timepoint == y.timepoint

        if one_is_control_or_blank() or not same_species():
            return 0

        else:
            rule = 'species'
            if same_donor():
                rule += '_&_donor'
            if same_family():
                rule += '_&_family'
            if same_timepoint():
                rule += '_&_timepoint'

        if rule in list(self.as_rules.keys()):
            return self.as_rules[rule]
        else:
            return 0

    @classmethod
    def from_rules(cls, rules):
        """Construct cost function from rules."""
        cfn = cls()
        cfn.costs = [rules[r] for r in cfn._rules]
        return cfn
    
    @property
    def costs(self):
        return -1 * self._costs

    @costs.setter
    def costs(self, vals):
        self._costs = np.array(vals)

    @property
    def as_rules(self):
        return {r: float(c) for r, c in zip(self._rules, self.costs)}


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