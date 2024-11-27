import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from xarray import DataArray

from .plates import Plate, PlateOccupancy


def plate_distance_matrix(plate: Plate):
    return _plate_da_distance(plate.data)


def _plate_da_distance(da: DataArray):
    coords = da.coords.to_index().to_frame().values
    return distance_matrix(coords, coords)


def prepare_design(sample_data, occupancy: PlateOccupancy):

    nonsample_design = (
        pd.DataFrame({
            'donor_id': (
                ['blank'] * occupancy.summary['num_blanks'] +
                ['control'] * occupancy.summary['num_controls']
            )
        })
        .assign(ID=np.nan, timepoint=np.nan, family=np.nan)
    )

    sample_design = (
        sample_data
        .filter(['ID', 'donor_id', 'timepoint', 'family'])
        
        # For viz
        .sort_values(['family', 'timepoint'])
    )

    design = pd.concat(
        [sample_design, nonsample_design], ignore_index=True
    )

    design['X1'] = pd.factorize(design['donor_id'], use_na_sentinel=True)[0]
    design['X2'] = pd.factorize(design['timepoint'], use_na_sentinel=True)[0]
    design['X3'] = pd.factorize(design['family'], use_na_sentinel=True)[0]

    return design


def samples_cost_matrix(sample_covars):

    num_samples = sample_covars.shape[0]

    cost_matrix = np.empty((num_samples, num_samples))

    for i in range(num_samples):
        for j in range(num_samples):
            cost_matrix[i, j] = cross_contamination_cost(
                sample_covars[i], sample_covars[j]
            )

    # add noise for stability
    cost_matrix = cost_matrix + np.random.normal(0, .1, cost_matrix.shape)

    return cost_matrix


def cross_contamination_cost(sample_a, sample_b):

    COSTS = np.array([0, 1, 2, 4, 10])
    
    # One is a control or blank
    if ((sample_a[1] == -1) or (sample_b[1] == -1)):
        return COSTS[0]
    
    # Each are from different donors
    elif sample_a[2] != sample_b[2]:
        return COSTS[0]
    
    # Same donor
    else:
        if np.array_equal(sample_a[:2], sample_b[:2]):
            return COSTS[4]
        
        elif sample_a[0] == sample_b[0]:
            return COSTS[3]

        elif sample_a[1] == sample_b[1]:
            return COSTS[2]
        
        else:
            return COSTS[1]