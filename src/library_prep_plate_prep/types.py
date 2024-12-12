from datetime import timedelta
from enum import StrEnum


class SampleCovars(StrEnum):
    SPECIES = 'species'
    DONOR = 'donor'
    FAMILY = 'family'
    TIMEPOINT = 'timepoint'


class Relationship(StrEnum):
    FATHER = 'Vater'
    CHILD = 'Kind'
    MOTHER = 'Mutter'
    BABY = 'Baby'


class Timepoint(StrEnum):
    BEFORE = 'vor'
    TWO_WEEKS = '2w'
    FOUR_WEEKS = '4w'
    TWO_MONTHS = '2m'
    THREE_MONTHS = '3m'
    SIX_MONTHS = '6m'
    NINE_MONTHS = '9m'
    TWELVE_MONTHS = '12m'

    def to_timedelta(self) -> timedelta:
        """Convert timepoint to timedelta for easy date calculations."""
        value = self.value.rstrip('w').rstrip('m')
        if 'vor' in self.value:
            return timedelta(days=-1)
        elif 'w' in self.value:
            return timedelta(weeks=int(value))
        elif 'm' in self.value:
            return timedelta(days=int(float(value) * (365 * 3 + 366) / (4 * 12)))