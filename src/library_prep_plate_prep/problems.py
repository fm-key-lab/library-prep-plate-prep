from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from library_prep_plate_prep import geometries

__all__ = ['QuadraticProblem', 'ArrangementProblem', 'soln_to_df']


class QuadraticProblem(ABC):
    def __init__(self, geom_xx: geometries.Geometry, geom_yy: geometries.Geometry):
        self._geom_xx = geom_xx
        self._geom_yy = geom_yy


class ArrangementProblem(QuadraticProblem):
    def __init__(self, plate: Union[geometries.Plate, geometries.Plates], samples: geometries.SequencingSamples):
        super().__init__(plate, samples)
        self._geom_plate = self._geom_xx
        self._geom_samples = self._geom_yy


def soln_to_df(problem: ArrangementProblem, solution: np.ndarray) -> pd.DataFrame:
    """Use solution to arrange plate and return as a DataFrame"""
    plates = problem._geom_plate
    plate_coords = plates.p_x_y

    samples = problem._geom_samples._data
    sample_ids = samples.index

    df = pd.DataFrame(
        plate_coords,
        columns=['plate', 'column', 'row'],
        index=pd.Index(sample_ids[solution], name='sample')
    )

    # Relabel: idx -> labels
    df['column'] = plates._plates[0].column_labels[df['column'].values - 1]
    df['row'] = plates._plates[0].row_labels[df['row'].values - 1]
    df['well'] = df['row'].astype(str) + df['column'].astype(str)
    
    return df