import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

__all__ = ['simulate_data']


def simulate_data(
    num_samples=50,
    num_donors=None,
    num_timepoints=None,
    num_families=None,
    random_state=None
):
    if num_samples < 20:
        raise ValueError(f'`num_samples` is {num_samples} but must be >= 20')
    
    num_donors = num_samples // 5 if num_donors is None else int(num_donors)
    num_timepoints = num_samples // 10 if num_timepoints is None else int(num_timepoints)
    num_families = num_samples // 20 if num_families is None else int(num_families)
    
    rng = np.random.default_rng(random_state)

    data = {}
    data['donor'] = rng.integers(1, num_donors, num_samples)
    data['timepoint'] = rng.integers(1, num_timepoints, num_samples)
    data['family'] = rng.integers(1, num_families, num_samples)

    sample_names = (
        'B' + np.strings.zfill(data['family'].astype(str), 3) + '_' 
        + np.strings.zfill(data['donor'].astype(str), 4) + ' @ ' 
        + np.strings.zfill(data['timepoint'].astype(str), 2) + ' days'
    )
    idx = pd.Index(sample_names, name='sample')    
    
    return pd.DataFrame(data, index=idx)