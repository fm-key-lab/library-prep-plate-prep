from abc import ABC, abstractmethod

from library_prep_plate_prep import geometries

__all__ = ['ArrangementProblem']


class QuadraticProblem(ABC):
    def __init__(self, geom_xx: geometries.Geometry, geom_yy: geometries.Geometry):
        self._geom_xx = geom_xx
        self._geom_yy = geom_yy


class ArrangementProblem(QuadraticProblem):
    def __init__(self, plate: geometries.Plate, samples: geometries.SequencingSamples):
        super().__init__(plate, samples)
        self._geom_plate = self._geom_xx
        self._geom_samples = self._geom_yy