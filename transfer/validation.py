"""Utilities for data validation."""
import logging
import pandas as pd
import pandera as pa
import itertools

logger = logging.getLogger()

COURSE_TYPES = ["LEC", "DIS", "SEM", "LAB", "others", "MISSING"]

# COURSE_COMBINED_TYPES contains the following:
# ['LEC-SEM', 'LEC-others', 'DIS-LEC', 'DIS-SEM', 'DIS-LAB', 'DIS-others',
# 'SEM-others', 'LAB-LEC', 'LAB-SEM', 'LAB-others']
COURSE_COMBINED_TYPES = [
    "-".join(c) for c in
    itertools.product(COURSE_TYPES, COURSE_TYPES)
    if c[0] < c[1]]

# MODALITIES = ["online", "inperson", "MISSING"]
SEXES = ["Male", "Female", "NotIndicated", "Other"]
ETHNICITIES = ['White', 'Asian', 'Black', '2 or More', 'Not Indic', 'Hispanic',
               'Hawaiian', 'Native Amr']
URM_STATUSES = ['Non-Underrepresented Minority', 'Underrepresented Minority',
                'International']
CIP2_CATEGORIES = ["{:02}".format(i) for i in range(1, 62)] + ["MISSING", ]

# Mapping of column names to allowed categorical values for each
# categorical column.
CATEGORICAL_FEATURE_ALLOWED_VALS = {
    # "modality": MODALITIES,
    "sex": SEXES,
    "ethnicity": ETHNICITIES,
    "urm_status": URM_STATUSES,
    "cip2_major_1": CIP2_CATEGORIES,
    "cip2_major_2": CIP2_CATEGORIES,
}

_SHARED_SCHEMA = {
    "units": pa.Column(float, pa.Check.in_range(0, 100)),
    "units_transferred": pa.Column(float, pa.Check.in_range(0, 100)),
    "units_failed": pa.Column(float, pa.Check.in_range(0, 100)),
    "units_incompleted": pa.Column(float, pa.Check.in_range(0, 100)),
    "units_withdrawn": pa.Column(float, pa.Check.in_range(0, 100)),
    "gpa_cumulative": pa.Column(float, pa.Check.in_range(0., 4.33)),
    "age": pa.Column(int, pa.Check.gt(0), coerce=True),
    "gpa_high_school": pa.Column(float, pa.Check.in_range(0., 5.),
                                 nullable=True),
    "act_english": pa.Column(float, pa.Check.in_range(0, 36), nullable=True),
    "act_math": pa.Column(float, pa.Check.in_range(0, 36), nullable=True),
    "sat_math": pa.Column(float, pa.Check.in_range(0, 800), nullable=True),
    "sat_verbal": pa.Column(float, pa.Check.in_range(0, 800), nullable=True),
    "gpa_avg": pa.Column(float, pa.Check.in_range(0., 4.33), nullable=True),
    "gpa_stddev": pa.Column(float, pa.Check.ge(0.), nullable=True),
    "gpa_zscore_avg": pa.Column(float, nullable=True),
    "gpa_zscore_stddev": pa.Column(float, pa.Check.ge(0.), nullable=True),
    **{"units_cip2_{}".format(cip): pa.Column(float, pa.Check.ge(0)) for cip in
       CIP2_CATEGORIES},
    **{"units_type_{}".format(course_type): pa.Column(float, pa.Check.ge(0)) for
       course_type in COURSE_TYPES + COURSE_COMBINED_TYPES},
    # **{"units_modality_{}".format(modality): pa.Column(float, pa.Check.ge(0))
    #    for modality
    #    in MODALITIES},
    "year": pa.Column(int, pa.Check.in_range(1990, 2022), coerce=True),
    "retention": pa.Column(bool, pa.Check.isin((True, False))),
    **{col: pa.Column(object, pa.Check.isin(allowed_vals))
       for col, allowed_vals in CATEGORICAL_FEATURE_ALLOWED_VALS.items()}
}

SCHEMA = pa.DataFrameSchema(_SHARED_SCHEMA)
COLNAMES = list(_SHARED_SCHEMA.keys())


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Check whether a df conforms to the expected schema, return if valid."""
    # First, check for the presence of exactly the expected columns
    data_cols = set(df.columns)
    schema_cols = set(COLNAMES)
    missing_cols = schema_cols - data_cols
    extra_cols = data_cols - schema_cols
    if len(missing_cols):
        msg = "data missing columns: {}".format(sorted(list(missing_cols)))
        logging.error(msg)
        raise ValueError(msg)
    if len(extra_cols):
        msg = "detected extra columns: {}".format(sorted(list(extra_cols)))
        logging.error(msg)
        raise ValueError(msg)
    # Check that the column conform to the expected schema
    validated = SCHEMA.validate(df)
    return validated


def prepare_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Prepares a one-hot encoding version of df given the schema constraints"""
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(categories=[SEXES, ETHNICITIES, URM_STATUSES],
                        drop="first", sparse=False)
    results = enc.fit_transform(df[["sex", "ethnicity", "urm_status"]])
    ndf = pd.DataFrame(results, columns=enc.get_feature_names_out())
    return df.drop(columns=["sex", "ethnicity", "urm_status"]).join(ndf)


def prepare_outcomes(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Splits dataset into X and y and labels retention data"""
    from sklearn.preprocessing import LabelEncoder
    y = pd.Series(LabelEncoder().fit_transform(df["retention"]))
    X = df.drop(columns="retention")
    return X, y
