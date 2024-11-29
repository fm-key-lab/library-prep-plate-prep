import pytest

from library_prep_plate_prep.arranger import PlateArranger


@pytest.fixture
def num_samples_range():
    return range(1, 200)


def test_plate_occupancy(num_samples_range):
    arranger = PlateArranger()
    for num_samples in num_samples_range:
        arranger._calc_occupancy(num_samples)