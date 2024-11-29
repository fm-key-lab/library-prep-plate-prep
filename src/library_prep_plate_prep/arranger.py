import copy
from typing import Optional, Union

import numpy as np

from library_prep_plate_prep import costs, plates, solvers
from library_prep_plate_prep.solvers import PlateAssignmentProblem


class PlateArranger:
    def __init__(
        self,
        plate_cost_fn: Optional[costs.PlateCostFn] = None,
        xcont_cost_fn: Optional[costs.CostFn] = None,
        solver: Optional[solvers.Solver] = None,
        plate: Optional[plates.Plate] = None,
    ):
        self.plate_cost_fn = costs.SqEuclidean() if plate_cost_fn is None else plate_cost_fn
        self.xcont_cost_fn = costs.CovarSimilarity() if xcont_cost_fn is None else xcont_cost_fn
        self.solver = solvers.QAP_2opt() if solver is None else solver
        self.plate = plates.Plate_96W() if plate is None else plate
        self.solution = {}
    
    def arrange(self, X, controls_per_plate: Optional[Union[float, int]] = 3, solver_kwargs={}):
        self._calc_occupancy(X.shape[0], controls_per_plate)
        self._provision_wells()
        self._add_blanks_controls(X)
        self._evaluate_cost_functions()
        self._solve(solver_kwargs)
        
        # TODO: Fix partial matching of controls
        
        # TODO: Arrange samples on plates
        
        return self.solution

    def _calc_occupancy(self, num_samples, controls_per_plate):
        
        def calc_controls():
            return int(np.ceil(
                controls_per_plate * num_samples
                 / (self.plate.wells - controls_per_plate)
            ))
        
        def calc_columns(num_controls):
            return int(
                self.plate.columns
                 * np.ceil((num_controls + num_samples) / self.plate.rows)
                 * self.plate.rows / self.plate.wells
            )

        def calc_blanks(num_columns, total_samples):
            return int(
                num_columns
                 * self.plate.rows
                 - total_samples
            )

        self.solution['controls'] = calc_controls()
        self.solution['columns'] = calc_columns(
            self.solution['controls']
        )
        self.solution['blanks'] = calc_blanks(
            self.solution['columns'],
            self.solution['controls'] + num_samples
        )
    
    def _provision_wells(self):
        num_plates = (
            - (
                self.solution['columns']
                 // -self.plate.columns
            )
        )
        
        def pseudofill_plate(num_columns):
            plate_copy = copy.deepcopy(self.plate)
            ds = plate_copy.data
            
            pseudofill = np.ones(
                (num_columns, plate_copy.rows),
                dtype='str'
            )
            ds['samples'].loc[{'x': range(num_columns)}] = pseudofill
            plate_copy.data = ds
            
            return plate_copy
        
        self.solution['plates'] = [
            pseudofill_plate(n) for n in
            map(
                lambda arr: int(arr.size),
                np.array_split(
                    np.ones(self.solution['columns']),
                    num_plates
                )
            )
        ]
    
    def _add_blanks_controls(self, X):
        num_nonsamples = (
            self.solution['blanks']
             + self.solution['controls']
        )
        self.solution['design'] = np.append(
            X,
            -np.ones((num_nonsamples, X.shape[1])),
            axis=0
        )
    
    def _evaluate_cost_functions(self):
        D = self.plate_cost_fn.all_pairs(
            self.solution['plates']
        )
        F = self.xcont_cost_fn.all_pairs(
            self.solution['design']
        )
        self.solution['prob'] = PlateAssignmentProblem(
            D, F
        )
        
    def _solve(self, kwargs):
        self.solution['P'] = self.solver(
            self.solution['prob'],
            **kwargs
        )