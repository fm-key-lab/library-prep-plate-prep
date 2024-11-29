from abc import ABC, abstractmethod

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial import distance_matrix

from library_prep_plate_prep.plates import Plate


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
    """Base class for costs over one or more plates."""

    def all_pairs(self, sample_data: np.ndarray) -> np.ndarray:
        return np.array([[self(x_, y_) for y_ in sample_data] for x_ in sample_data])

    @abstractmethod
    def __call__(self, x, y) -> float:
        pass


class CovarSimilarity(SamplesCostFn):
    """Sample covariate similarity."""
    
    COSTS = np.array([0, 1, 2, 4, 10]) * -1
  
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        
        def one_is_control_or_blank():
            return (x[1] == -1) or (y[1] == -1)
        
        def same_donor():
            return x[0] == y[0]
        
        def same_timepoint():
            return x[1] == y[1]

        def same_family():
            return x[2] == y[2]

        if one_is_control_or_blank():
            return self.COSTS[0]

        elif not same_family():
            return self.COSTS[0]
        
        else:
            if same_donor() and same_timepoint():
                return self.COSTS[4]
            
            elif same_donor():
                return self.COSTS[3]

            elif same_timepoint():
                return self.COSTS[2]

            else:
                return self.COSTS[1]
  

class PlateCostFn(CostFn):
    """Base class for costs over one or more plates."""

    def all_pairs(self, plates: list[Plate]) -> np.ndarray:
        return block_diag(*[self(plate.x_y) for plate in plates])

    @abstractmethod
    def __call__(self, x, y) -> float:
        pass


class SqEuclidean(PlateCostFn):
    """Squared Euclidean distance."""

    def __call__(self, z):
        return distance_matrix(z, z)