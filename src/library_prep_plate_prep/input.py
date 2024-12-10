import pandera as pa

from library_prep_plate_prep.types import Relationship, Timepoint

timepoint_key


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
        return (
            s
            .apply(Timepoint)
            .apply(lambda x: x.to_timedelta().days)
        )