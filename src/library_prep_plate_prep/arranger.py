import copy
from typing import Optional, Union

import numpy as np
import pandas as pd

from library_prep_plate_prep import costs, plates, solvers
from library_prep_plate_prep.solvers import PlateAssignmentProblem


class PlateArranger:
    def __init__(
        self,
        plate_cost_fn: Optional[costs.PlateCostFn] = None,
        xcont_cost_fn: Optional[costs.CostFn] = None,
        ctrl_sampler: Optional[solvers.Sampler] = None,
        solver: Optional[solvers.Solver] = None,
        plate: Optional[plates.Plate] = None,
    ):
        self.solution = {}
        self.plate_cost_fn = costs.SqEuclidean() if plate_cost_fn is None else plate_cost_fn
        self.xcont_cost_fn = costs.CovarSimilarity() if xcont_cost_fn is None else xcont_cost_fn
        self.ctrl_sampler = solvers.LHSampler_CenterMaximin() if ctrl_sampler is None else ctrl_sampler
        self.solver = solvers.QAP_2opt() if solver is None else solver
        self.plate = plates.Plate_96W() if plate is None else plate
    
    def arrange(
        self,
        X: pd.DataFrame,
        controls_per_plate: Optional[Union[float, int]] = 3,
        sampler_kwargs: Optional[dict] = None,
        solver_kwargs: Optional[dict] = None,
        return_df: Optional[bool] = False,
    ):
        self._calc_occupancy(X.shape[0], controls_per_plate)
        self._provision_wells()
        self._add_blanks_controls_to_design(X)
        self._evaluate_cost_functions()
        self._seed_controls({} if sampler_kwargs is None else sampler_kwargs)
        self._solve({} if solver_kwargs is None else solver_kwargs)
        self._arrange_samples()
        
        if return_df:
            return (
                pd.concat(
                    [plate.as_df() for plate in self.solution['plates']],
                    ignore_index=True,
                )
                .set_index(['column', 'row'])
            )

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
        
        def pseudofill_plate(num_columns):
            plate_copy = copy.deepcopy(self.plate)
            ds = plate_copy.data
            
            pseudofill = np.ones(
                (num_columns, plate_copy.rows),
                dtype='str'
            )
            ds['samples'].loc[{'column': range(1, num_columns + 1)}] = pseudofill
            plate_copy.data = ds
            
            return plate_copy
        
        num_plates = (
            - (
                self.solution['columns']
                 // -self.plate.columns
            )
        )
        
        self.solution['columns'] = list(
            map(
                lambda arr: int(arr.size),
                np.array_split(
                    np.ones(self.solution['columns']),
                    num_plates
                )
            )
        )
        
        self.solution['controls'] = list(
            map(
                lambda arr: int(arr.size),
                np.array_split(
                    np.ones(self.solution['controls']),
                    num_plates
                )
            )
        )
        
        self.solution['plates'] = [
            pseudofill_plate(n) for n in self.solution['columns']
        ]
    
    def _add_blanks_controls_to_design(self, X):
        num_nonsamples = (
            self.solution['blanks']
             + sum(self.solution['controls'])
        )
        
        # NOTE: Fix input variables for blanks/controls to -1
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

    def _seed_controls(self, kwargs):
        ctrl_seeds = []

        num_samples = self.solution['design'].shape[0]
        last_sample = num_samples
        
        for num_controls, num_columns, plate in zip(
            self.solution['controls'],
            self.solution['columns'],
            self.solution['plates'],
        ):
            
            def seed_single_plate():
                ctrl_x_y = self.ctrl_sampler.seed_control_wells(
                    num_controls, num_columns, self.plate.rows, **kwargs
                )
                
                def get_plate_idx():
                    return np.array([
                        np.where((plate.x_y == coord).all(axis=1))[0] for coord in ctrl_x_y
                    ]).flatten()
                
                def get_sample_idx():
                    return np.arange(last_sample - num_controls, last_sample)
                
                return np.c_[get_plate_idx(), get_sample_idx()]

            ctrl_seeds.append(seed_single_plate())
            last_sample -= num_controls

        self.solution['seed'] = np.concatenate(ctrl_seeds)
        
    def _solve(self, kwargs):
        
        def update_kwargs_w_seed():            
            """Update seed to add controls."""
            if 'partial_match' in list(kwargs.keys()):
                kwargs['partial_match'] = np.concatenate([kwargs['partial_match'], self.solution['seed']])
            else:
                kwargs['partial_match'] = self.solution['seed']
            
            return kwargs
        
        kwargs_w_seed = update_kwargs_w_seed()
        
        self.solution['P'] = self.solver(
            self.solution['prob'],
            **kwargs_w_seed
        )
    
    def _arrange_samples(self):
        plate_x_y = np.vstack([
            np.hstack([np.full(plate.x_y.shape[0], i)[:, None], plate.x_y]) 
            for i, plate in enumerate(self.solution['plates'])
        ])

        # ------------------------
        # TODO: Temporary
        sample_ids = np.char.add(np.array(['sample_']), np.arange(self.solution['design'].shape[0]).astype(str))
        # ------------------------
        
        sample_ids_soln = sample_ids[self.solution['P']]

        for i, plate in enumerate(self.solution['plates']):
            pxy = plate_x_y[plate_x_y[:, 0] == i][:, 1:]
            sid = sample_ids_soln[plate_x_y[:, 0] == i]

            plate_data = plate.data.copy(deep=True)            
            
            for j, (x, y) in enumerate(pxy):
                plate_data['samples'].loc[{
                    'column': plate.xy_to_cr['column'][x],
                    'row': plate.xy_to_cr['row'][y]
                }] = str(sid[j])
            
            plate.data = plate_data