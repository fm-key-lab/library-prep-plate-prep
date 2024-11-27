import string
from dataclasses import dataclass, field

import numpy as np
import xarray as xr


@dataclass
class PlateOccupancy:
    num_samples: int
    num_blanks: int = field(init=False)
    num_controls: int = field(init=False)
    num_rows: int = field(init=False)
    num_columns: int = field(init=False)
    num_plates: int = field(init=False)
    controls_per_plate: int = 3
    shape: np.ndarray = field(default_factory=lambda: np.array([12, 8])) # columns x rows

    @property
    def summary(self) -> dict[str, int]:
        return {
            'num_samples': self.num_samples,
            'num_rows': self.num_rows,
            'num_controls': self.num_controls,
            'num_blanks': self.num_blanks,
            'num_columns': self.num_columns,
            'num_plates': self.num_plates,
        }

    def required_controls(self):
        # TODO: Enforce a minimum number of control wells per plate?
        return np.ceil(self.controls_per_plate * self.num_samples / (self.shape.prod() - self.controls_per_plate))

    def required_plates(self, total_samples):
        '''Fill each plate column completely (max volume, min blanks).'''
        samples_per_col = self.shape[1] # Num. of wells in a column equal to num. rows
        columns = np.ceil(total_samples / samples_per_col)
        return columns * samples_per_col / self.shape.prod()

    def remaining_wells(self):
        pass

    def __post_init__(self):
        tot_columns, tot_rows = self.shape
        
        # NOTE: Fill entire columns (so all rows used)
        self.num_rows = int(tot_rows)

        num_controls = self.required_controls()

        # NOTE: Handles edge cases where filling out columns (rounding up) -> 
        #       more controls needed than `num_controls`.
        #       (Unsure when such cases exist...)
        fractional_plates = 0 # init
        adj = 0
        while np.ceil(self.controls_per_plate * fractional_plates) != num_controls:
            fractional_plates = self.required_plates(self.num_samples + num_controls)
            num_controls += adj
            adj += 1
        
        self.num_controls = int(num_controls)
        self.num_columns = int(tot_columns * fractional_plates) # Num. of wells in a column equal to num. rows
        self.num_plates = int(np.ceil(fractional_plates))
        self.num_blanks = int(self.shape.prod() * fractional_plates - (self.num_samples + self.num_controls))


class Plate:
    def __init__(self, shape):
        self._values = np.zeros(shape)
        self.data = None
        self.shape = shape # columns x rows
        self._init()

    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, values):
        self._values = values

    def _init(self):
        labels = make_plate_labels(*self.shape)

        # NOTE: Physical coordinates on unit interval
        self.data = xr.DataArray(
            data=self.values,
            coords={
                'x': np.arange(self.shape[0]),
                'y': np.arange(self.shape[1]),
                'column': ('x', labels[0]),
                'row': ('y', labels[1]),
                'well': (('x', 'y'), labels[2]),
            },
            dims=['x', 'y']
        )

    def __str__(self):
        return f'Plate({self.shape[0]} columns, {self.shape[1]} rows)'
    
    def __repr__(self):
        return str(self)


def make_plate_labels(num_columns, num_rows) -> list[np.ndarray, np.ndarray, np.ndarray]:

    def columns():
        return np.arange(num_columns) + 1
    
    def rows():
        return np.array(list(string.ascii_uppercase[:num_rows]))

    column_labels, row_labels = columns(), rows()

    def wells():
        return np.char.add(
            *np.broadcast_arrays(
                row_labels[None, :],
                column_labels[:, None].astype(str)
            )
        )
    
    well_labels = wells()
    
    return column_labels, row_labels, well_labels