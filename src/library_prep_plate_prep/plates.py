import string
from dataclasses import dataclass

import numpy as np
import xarray as xr


class Plate:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows
        self.wells = columns * rows
        self._samples = np.zeros((columns, rows), dtype='<U20')
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
            [['x', 'y']]
            .to_array()
            .to_numpy()
            .reshape(2, -1)
            .T
        )

    @property
    def xy_to_cr(self):
        return {
            d: {
                k: v for k, v in zip(
                    self.data[c].to_dict()['data'],
                    self.data[c].to_dict()['coords'][d]['data']
                )
            } for c, d in zip(['x', 'y'], ['column', 'row'])
        }

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
        return xr.Dataset(
            {
                'occupied': (['column', 'row'], self.occupied),
                'samples': (['column', 'row'], self.samples),
                
                # NOTE: Physical coordinates on unit intervals
                'x': ('column', np.arange(self.columns)),
                'y': ('row', np.arange(self.rows)),
            },
            coords=self._make_plate_labels(),
        )

    def as_df(self):
        return (
            self.data
            ['samples']
            .to_pandas()
            .melt(ignore_index=False, value_name='sample')
            .reset_index()
            .assign(sample=lambda df: df['sample'].replace('', np.nan))
            .dropna()
            .sort_values(['column', 'row'])
        )

    def _make_plate_labels(self):

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
        
        coords = {}
        coords['column'] = make_column_labels()
        coords['row'] = make_row_labels()
        coords['well'] = (
            ('column', 'row'),
            make_well_labels(coords['column'], coords['row'])
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