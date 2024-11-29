import string
from dataclasses import dataclass

import numpy as np
import xarray as xr


class Plate:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.wells = columns * rows
        self._samples = np.zeros((columns, rows), dtype='str')
        self._data = self.as_xr_dataset()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: xr.Dataset):
        self.samples = data['samples'].values
        self._data = self.as_xr_dataset()

    @property
    def x_y(self):        
        return (
            self.data
            .where(self.data['occupied'], drop=True)
            .coords
            .to_index()
            .to_frame()
            .values
        )

    @property
    def samples(self):
        return self._samples
    
    @samples.setter
    def samples(self, samples):
        self._samples = samples

    @property
    def occupied(self):
        return self.samples != ''

    def as_xr_dataset(self):
        plate_coords = {}

        # NOTE: Physical coordinates on unit intervals
        plate_coords['x'] = np.arange(self.columns)
        plate_coords['y'] = np.arange(self.rows)

        # Plate label coordinates
        plate_coords = self._make_plate_labels(plate_coords)

        # TODO: Should use (columns, rows) as coords (not (x, y))
        return xr.Dataset(
            {
                'occupied': (['x', 'y'], self.occupied),
                'samples': (['x', 'y'], self.samples),
            },
            coords=plate_coords,
        )

    def _make_plate_labels(self, coords: dict):

        def make_column_labels():
            return np.arange(self.columns) + 1
        
        def make_row_labels():
            return np.array(list(string.ascii_uppercase[:self.rows]))

        def make_well_labels(column_labels, row_labels):
            return np.char.add(
                *np.broadcast_arrays(
                    row_labels[None, :],
                    column_labels[:, None].astype(str)
                )
            )
        
        coords['columns'] = ('x', make_column_labels())
        coords['rows'] = ('y', make_row_labels())
        coords['wells'] = (
            ('x', 'y'), 
            make_well_labels(coords['columns'][1], coords['rows'][1])
        )
        
        return coords

    def __deepcopy__(self, memo):
        self.data = self.data.copy(deep=True)
        return self

@dataclass
class Plate_96W(Plate):
    columns: int = 12
    rows: int = 8

    def __post_init__(self):
        super().__init__(self.columns, self.rows)
