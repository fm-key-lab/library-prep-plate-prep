import numpy as np

__all__ = ['calc_init_vals', 'idx_plate']


def idx_plate(a: np.ndarray, c: int):
    return np.pad(a, ((0,0), (1,0)), constant_values=c)


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