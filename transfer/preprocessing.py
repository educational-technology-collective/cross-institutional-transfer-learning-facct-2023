"""
Shared utilities for preprocessing (validated) data.
"""
import logging
from typing import Any, List

import pandas as pd
from transfer.validation import CATEGORICAL_FEATURE_ALLOWED_VALS

# Drop observations which contain missing values in these columns.
DROP_MISSING_COLUMNS = [
    'gpa_avg',
    'gpa_stddev',
    'gpa_zscore_avg',
    'gpa_zscore_stddev',
    'gpa_high_school',
    'act_english',
    'act_math', ]

# IMPUTE observations which contain missing values in these columns.
IMPUTE_MISSING_COLUMNS = ['sat_math', 'sat_verbal']


def get_columns_matching_dtype(df: pd.DataFrame, dtype: str) -> List:
    return [c for c in df.columns if df[c].dtype == dtype]


def get_numeric_columns(df):
    return get_columns_matching_dtype(df, int) + get_columns_matching_dtype(df, float)


def make_dummy_colname(colname: str, val: Any) -> str:
    return f"{colname}_{val}"


def bool_to_indicator(df: pd.DataFrame) -> pd.DataFrame:
    # Convert boolean cols into 0/1
    boolean_cols = get_columns_matching_dtype(df, 'bool')
    for colname in boolean_cols:
        df[colname] = df[colname].astype(int)
    return df


def create_dummies(df: pd.DataFrame) -> pd.DataFrame:
    # Convert object cols into dummies
    categorical_cols = get_columns_matching_dtype(df, 'object')
    for colname in categorical_cols:
        coldata = df.pop(colname)
        logging.debug("processing column %s with %s unique values",
                      colname, len(coldata.unique()))
        dummies = pd.get_dummies(coldata, prefix=colname, dummy_na=True)

        # Ensure a dummy column is created for every distinct possible
        # value, even if it is not present in the data.
        for val in CATEGORICAL_FEATURE_ALLOWED_VALS[colname]:
            dummy_colname = make_dummy_colname(colname, val)
            if dummy_colname not in dummies.columns:
                logging.debug(
                    "column %s not in data; creating dummy" % dummy_colname)
                dummies[dummy_colname] = 0
        df = pd.concat((df, dummies), axis=1)
    return df


def preprocess_fn(df: pd.DataFrame, convert_categorical_to_numeric=True):
    """Preprocess a dataframe of validated data.

    This function implements the shared logic that should be applied
        to all institutions' data after it has passed validation.

    Args:
        df: the dataframe to process.
        convert_categorical_to_numeric: if True, convert all categorical
            columns to numeric dummy variables; else leave as-is.
    Return:
        A copy of the dataframe after the resulting operations.
    """
    if convert_categorical_to_numeric:
        df = create_dummies(df)
        df = bool_to_indicator(df)

    df = df.dropna(axis=0, subset=DROP_MISSING_COLUMNS)
    for column_to_impute in IMPUTE_MISSING_COLUMNS:
        df[column_to_impute] = df[column_to_impute].fillna(
            df[column_to_impute].mean())
    # Sort columns lexicographically
    df = df.reindex(sorted(df.columns), axis=1)
    return df
