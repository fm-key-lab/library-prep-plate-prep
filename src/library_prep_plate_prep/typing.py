from enum import StrEnum


class Relationship(StrEnum):
    FATHER = 'Vater'
    CHILD = 'Kind'
    MOTHER = 'Mutter'
    BABY = 'Baby'


class Timepoint(StrEnum):
    BEFORE = 'vor'
    TWO_WEEKS = '2W'
    FOUR_WEEKS = '4W'
    TWO_MONTHS = '2M'
    THREE_MONTHS = '3M'
    SIX_MONTHS = '6M'
    NINE_MONTHS = '9M'
    TWELVE_MONTHS = '12M'