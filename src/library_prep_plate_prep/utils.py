import numpy as np
import pandas as pd

from library_prep_plate_prep.problems import ArrangementProblem

__all__ = ['soln_to_df', 'calc_init_vals']


def soln_to_df(problem: ArrangementProblem, solution: np.ndarray) -> pd.DataFrame:
    """Use solution to arrange plate and return as a DataFrame"""
    plates = problem._geom_plate
    plate_coords = plates.p_x_y

    samples = problem._geom_samples._data
    sample_ids = samples.index

    return pd.DataFrame(
        plate_coords,
        columns=['plate', 'column', 'row'],
        index=pd.Index(sample_ids[solution], name='sample')
    )


def calc_init_vals(n_samples, n_ctrls_plate=3, n_plate_cols=12, n_plate_rows=8):
    """Provision plates.

    Assumptions:
        - Use every row.
        - Fill every column completely.

    """
    n_plate_wells = n_plate_cols * n_plate_rows

    def tot_controls():
        return int(np.ceil(n_ctrls_plate * n_samples / (n_plate_wells - n_ctrls_plate)))

    tot_ctrls = tot_controls()

    def tot_columns():
        return int(
            n_plate_cols
            * np.ceil((tot_ctrls + n_samples) / n_plate_rows)
            * n_plate_rows / n_plate_wells
        )

    tot_cols = tot_columns()

    def tot_blanks():
        return int(tot_cols * n_plate_rows - (tot_ctrls + n_samples))

    tot_empty = tot_blanks()
    tot_plates = -(tot_cols // -n_plate_cols)

    def tot_rows():
        return [n_plate_rows] * tot_plates

    return {
        'n_controls': _div_among_plates(tot_ctrls, tot_plates),
        'n_columns': _div_among_plates(tot_cols, tot_plates),
        'n_rows': tot_rows(),
        'n_empty': tot_empty
    }


def _div_among_plates(tot, n_plates):
    x = np.array_split(np.ones(tot), n_plates)
    return list(map(lambda arr: int(arr.size), x))