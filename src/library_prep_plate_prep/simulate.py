import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

__all__ = ['simulate_data']


def simulate_data(
    n_samples=50,
    n_species=None,
    n_donor=None,
    n_family=None,
    n_timepoint=None,
    random_state=None
):
    if n_samples < 20:
        raise ValueError(f'`n_samples` is {n_samples} but must be >= 20')
    
    rng = np.random.default_rng(random_state)

    MAX_DONORS = 4
    MAX_SPECIES = 5
    VARS = ['species', 'donor', 'family', 'timepoint']
    
    sim_cfg = {
        'n_species': min(MAX_SPECIES, n_samples // 20 if n_species is None else int(n_species)),
        'n_donor': min(MAX_DONORS, n_samples // 5 if n_donor is None else int(n_donor)),
        'n_family': n_samples // 20 if n_family is None else int(n_family),
        'n_timepoint': n_samples // 10 if n_timepoint is None else int(n_timepoint),
    }
    
    data = {l.split('n_')[1]: rng.integers(1, n, n_samples) for l, n in sim_cfg.items()}

    sample_names = (
        'species ' + data['species'].astype(str) + ':'
        + 'B' + np.strings.zfill(data['family'].astype(str), 3) + '_' 
        + np.strings.zfill(data['donor'].astype(str), 4) + ' @ ' 
        + np.strings.zfill(data['timepoint'].astype(str), 2) + ' days'
    )
    idx = pd.Index(sample_names, name='sample')
    
    return pd.DataFrame(data, index=idx)