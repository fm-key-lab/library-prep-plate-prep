import numpy as np
import pandas as pd


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
    num_families = num_samples // 5 if num_families is None else int(num_families)
    
    rng = np.random.default_rng(random_state)

    data = {}
    data['donor'] = rng.integers(1, num_donors, num_samples)
    data['timepoint'] = rng.integers(1, num_timepoints, num_samples)
    data['family'] = rng.integers(1, num_families, num_samples)

    design = pd.DataFrame(data, index=pd.Index(range(num_samples), name='sample'))
    
    return design