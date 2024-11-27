import pytest

from library_prep_plate_prep.plates import PlateOccupancy


@pytest.fixture
def num_samples_range():
    return range(1, 200)


def test_plate_occupancy(num_samples_range):
    for num_samples in num_samples_range:
        PlateOccupancy(num_samples)