from enum import StrEnum

import pandera as pa


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


class LibrarySchema(pa.DataFrameModel):
    ini_library: str
    ini_plate: str
    ini_well: str
    ini_lib_conc: str


class LibPrepDataSchema(SampleInfoSchema, LibrarySchema):
    relationship: str = pa.Field(isin=list(Relationship))