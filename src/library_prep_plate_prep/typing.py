from enum import StrEnum

import pandas as pd
import pandera as pa

pd.set_option('future.no_silent_downcasting', True)


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
    

class SampleInfoSchema(pa.DataFrameModel):
    ID: str = pa.Field(str_matches=r'^B\d\d\d_\d\d\d\d$')
    donor_id: str
    family: str = pa.Field(str_matches=r'^B\d\d\d$')
    relationship: str
    timepoint: str = pa.Field(isin=list(Timepoint))

    @pa.parser('relationship')
    def clean(cls, s):
        return (
            s.str
            .extract(f'({"|".join(list(Relationship))})')
            .get(0)
            .rename('relationship')
        )


class SourcePlateSchema(pa.DataFrameModel):
    ini_library: str
    ini_plate: str
    ini_well: str
    ini_lib_conc: str


class LibPrepDataSchema(SampleInfoSchema, SourcePlateSchema):
    relationship: str = pa.Field(isin=list(Relationship))
    timepoint: int
    
    @pa.parser('timepoint')
    def timepoint_to_days(cls, s):

        TIMEPOINT_KEY = {
            'vor': -1,
            '2W': 14,
            '4W': 28,
            '2M': 61,
            '3M': 91,
            '6M': 183,
            '9M': 271,
            '12M': 365,
        }
        
        return s.replace(TIMEPOINT_KEY).astype(int)