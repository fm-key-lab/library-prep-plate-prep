import copy
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

import numpy as np
import pandas as pd

from library_prep_plate_prep import costs, solvers
from library_prep_plate_prep.plates import Plate, Plate_96W


class PlateArranger:
    """The plate arranger."""

    plate = Plate_96W()
    
    def __init__(
        self,
        plate_cost_fn: Optional[costs.PlateCostFn] = None,
        xcont_cost_fn: Optional[costs.CostFn] = None,
        ctrl_sampler: Optional[solvers.Sampler] = None,
        solver: Optional[solvers.Solver] = None,
    ):
        self.solution = {}
        self.plate_cost_fn = costs.SqEuclidean() if plate_cost_fn is None else plate_cost_fn
        self.xcont_cost_fn = costs.CovarSimilarity() if xcont_cost_fn is None else xcont_cost_fn
        self.ctrl_sampler = solvers.LHSampler_CenterMaximin() if ctrl_sampler is None else ctrl_sampler
        self.solver = solvers.QAP_2opt() if solver is None else solver
    
    def arrange(
        self,
        X: pd.DataFrame,
        partial_solution: Optional[dict] = None,
        controls_per_plate: Optional[Union[float, int]] = 3,
        sampler_kwargs: Optional[dict] = None,
        solver_kwargs: Optional[dict] = None,
        return_df: Optional[bool] = False,
    ):
        self._init_soln(X, partial_solution, controls_per_plate)
        self._seed_controls({} if sampler_kwargs is None else sampler_kwargs)
        self._solve({} if solver_kwargs is None else solver_kwargs)
        
        if return_df:
            return arrangement_to_df(self.solution['plates'])

    def _init_soln(self, X, psoln, ctrls_pp):
        """Skeleton arrangement."""
        p = _PartialArrangement(**psoln) if psoln else _PartialArrangement.from_n(X.shape[0], ctrls_pp)
        self.solution.update(asdict(p))
        self.solution['plates'] = p.plates
        self.solution['rows'] = [self.plate.rows] * len(p.plates)
        
        # Update design
        n_nonsamp = self.solution['empties'] + sum(self.solution['controls'])
        X_nonsamp = -np.ones((n_nonsamp, X.shape[1]))
        self.solution['design'] = np.append(X, X_nonsamp, axis=0)
    
    def _seed_controls(self, kwargs):
        """Arrange control wells."""
        self.solution['partial_match'] = seed_controls(
            self.solution['design'].shape[0],
            self.solution['controls'],
            self.solution['columns'],
            self.solution['rows'],
            self.solution['plates'],
            self.ctrl_sampler,
            **kwargs
        )
    
    def _solve(self, kwargs):
        """Solve."""
        self.solution['D'] = self.plate_cost_fn.all_pairs(self.solution['plates'])
        self.solution['F'] = self.xcont_cost_fn.all_pairs(self.solution['design'])
        self.solution['P'] = self.solver(self.solution['D'], self.solution['F'], **kwargs)
        self._arrange_samples()
    
    def _arrange_samples(self):
        """Translate solution to plate(s)."""
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


def seed_controls(n, ctrls, cols, rows, plates, sampler, **kwargs):
    ctrl_seeds = []
    for ctrl, col, row, plate in zip(ctrls, cols, rows, plates):
        def seed_single_plate():
            ctrl_x_y = sampler.seed_control_wells(ctrl, col, row, **kwargs)
            
            def get_plate_idx():
                return np.array([np.where((plate.x_y == coord).all(axis=1))[0] for coord in ctrl_x_y]).flatten()
            
            def get_sample_idx():
                return np.arange(n - ctrl, n)
            
            return np.c_[get_plate_idx(), get_sample_idx()]

        ctrl_seeds.append(seed_single_plate())
        n -= ctrl
    
    return np.concatenate(ctrl_seeds)


def arrangement_to_df(plates):
    dfs = [plate.as_df() for plate in plates]
    return pd.concat(dfs, ignore_index=True).set_index(['column', 'row'])


@dataclass
class _PartialArrangement:
    """Partial solution to plate arrangement."""

    plate = Plate_96W()
    columns: Union[int, list[int]]
    controls: Union[int, list[int]]
    empties: Union[int, list[int]]
    plates: list[Plate] = field(init=False)
    
    @classmethod
    def from_n(cls, n, ctrls_pp=3):
        ctrls, cols, empty = calc_plate_occupancy(
            cls.plate, n, ctrls_pp
        )
        plates = -(cols // -cls.plate.columns)
        psoln = cls(*[div_among_plates(_, plates) for _ in [cols, ctrls]], empty)
        return psoln

    def __post_init__(self):
        self.plates = self.pseudofill()

    def pseudofill(self):
        def pseudofill_plate(n):
            plate_copy = copy.deepcopy(self.plate)
            ds = plate_copy.data
            
            pseudofill = np.ones((n, plate_copy.rows), dtype='str')
            ds['samples'].loc[{'column': range(1, n + 1)}] = pseudofill

            plate_copy.data = ds            
            return plate_copy

        return [pseudofill_plate(cols) for cols in self.columns]


def calc_plate_occupancy(plate: Plate, num_samples: int, ctrls_pp: int):
    def calc_controls():
        return int(np.ceil(ctrls_pp * num_samples / (plate.wells - ctrls_pp)))
    
    def calc_columns(num_ctrls):
        return int(
            plate.columns
                * np.ceil((num_ctrls + num_samples) / plate.rows)
                * plate.rows / plate.wells
        )

    def calc_blanks(num_cols, total_samples):
        return int(num_cols * plate.rows - total_samples)

    controls = calc_controls()
    columns = calc_columns(controls)
    blanks = calc_blanks(columns, controls + num_samples)

    return controls, columns, blanks


def div_among_plates(m, n):
    x = np.array_split(np.ones(m), n)
    return list(map(lambda arr: int(arr.size), x))