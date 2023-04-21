"""
Utilities for running experiments.
"""
import logging
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from transfer import keys
from transfer.keys import MODEL_DIR, METRICS_DIR, TARGET_COLNAME, \
    INSTITUTION_CODES, MLP, LIGHTGBM, L2LR, RF
from transfer import preprocessing

RANDOM_STATE = 4620119

# Grid of l2 lambda values to sweep over; this is same range as default
# scikit-learn grid for LogisticRegressionCV.
LAMBDA_GRID = [0., 1e-4, 1e-3, 1e-2, 1e-1, 1., 10, 100, 1000, 10000]

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

# Default values to (optionally) override the library default values
# for each model. These are used for the base models, as well as the
# ensemble models.
DEFAULT_MODEL_KWARGS = {
    MLP: {'num_layers': 2,
          'd_hidden': 512,
          'l2lambda': 0.},
    LIGHTGBM: {},
    L2LR: {},
    RF: {},
}

HPARAM_GRID = {
    # kwargs passed to the model constructor fn prefixed by 'model__'
    MLP: {
        'model__d_hidden': [256, 512, 1024],
        'model__num_layers': [1, 2, 3],
        'batch_size': [256],
        'epochs': [50]},
    L2LR: None,
    LIGHTGBM: [{'n_estimators': [100, 250, 500],
                'num_leaves': [31, 63, 127],
                'min_child_samples': [5, 10, 20],
                'colsample_bytree': [1., 0.9, 0.8]}],
}


def make_uid(**kwargs) -> str:
    """Helper function to build a unique identifier for an experiment."""
    # TODO(jpgard): this is a hack; do a better job of this.
    uid = "_".join([str(k) + str(v) for k, v in sorted(kwargs.items())])
    return uid


def _initialize_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return


def get_model_path(uid: str, suffix: str) -> str:
    filename = f"{uid}{suffix}"
    filepath = os.path.join(MODEL_DIR, filename)
    _initialize_dir(MODEL_DIR)
    return filepath


def get_metrics_path(uid: str, extension: str = "csv") -> str:
    filename = f"{uid}.{extension}"
    filepath = os.path.join(METRICS_DIR, filename)
    _initialize_dir(METRICS_DIR)
    return filepath


def write_description(institution, split, df: pd.DataFrame, scaled: bool,
                      interaction_degree: Optional[int] = None):
    if (interaction_degree is None) or (interaction_degree < 2):
        filename = f"{institution}_{split}_descriptivestats_scaled{scaled}.csv"
    else:
        filename = f"{institution}_{split}_descriptivestats_ord{interaction_degree}scaled{scaled}.csv"
    filepath = os.path.join(METRICS_DIR, filename)
    _initialize_dir(METRICS_DIR)
    logging.info("writing data description to %s", filepath)
    df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9],
                include='all').reset_index().rename(
        columns={"index": "metric"}).to_csv(filepath, index=False)


def create_interactions(df, degree: int, target_colname=keys.TARGET_COLNAME):
    """Utility function to create interactions for the non-target variables."""
    if degree > 2:
        logging.warning("using degree > 2 can create huge output DataFrames;"
                        "may OOM.")
    logging.info(
        "computing polynomial features of degree %s from %s input features",
        degree, df.shape[1])
    df_x, df_y = x_y_split(df, target_colname)
    input_feature_names = df_x.columns
    pf = PolynomialFeatures(degree=degree)
    df_x = pf.fit_transform(df_x)
    output_feature_names = pf.get_feature_names(input_feature_names)
    df = pd.DataFrame(df_x, columns=output_feature_names)
    df[target_colname] = df_y.values  # re-insert the targets
    assert pd.isnull(df).sum().sum() == 0, "null values detected in data."
    return df


def apply_scaler_to_x(df: pd.DataFrame,
                      scaler: sklearn.preprocessing.StandardScaler):
    """Apply scaler to numeric features."""
    colnames_scaler = [c for c in scaler.feature_names_in_]
    colnames_unscaled = [c for c in df.columns if c not in colnames_scaler]
    scale_df = df[colnames_scaler]
    x_scaled = scaler.transform(scale_df)

    # Cast back to DataFrame and assign columns because StandardScaler
    # converts to np.ndarray.
    df_out = pd.DataFrame(x_scaled, columns=list(scaler.feature_names_in_))
    # Copy over the original data from unscaled columns
    for unscaled_colname in colnames_unscaled:
        # use .values to avoid type-casting in assignment
        df_out[unscaled_colname] = df[unscaled_colname].values
    assert set(df_out.columns) == set(df.columns)
    assert np.all((df_out[df.columns].dtypes == df.dtypes).values)
    return df_out


def scale_data(train_df: pd.DataFrame, test_df: pd.DataFrame,
               validation_df: pd.DataFrame, eval_df: pd.DataFrame,
               target_colname=keys.TARGET_COLNAME,
               unscaled_numeric_cols=("year")):
    """Scale data to mean 1 variance 1 using the training data."""
    # Fit the scaler on the (non-label) features of the training data.
    scaler = sklearn.preprocessing.StandardScaler()
    colnames_to_scale = [x for x in preprocessing.get_numeric_columns(train_df)
                         if
                         x != target_colname and x not in unscaled_numeric_cols]
    scaler.fit(train_df[colnames_to_scale])

    train_df = apply_scaler_to_x(train_df, scaler)
    test_df = apply_scaler_to_x(test_df, scaler)
    validation_df = apply_scaler_to_x(validation_df, scaler)
    eval_df = apply_scaler_to_x(eval_df, scaler)

    return train_df, test_df, validation_df, eval_df


def read_and_split_data(
        institution: str,
        preprocessed_data_dir: str,
        describe=True) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read institution data, preprocess it, and split into train/test/valid/eval.

    :param institution: string name of institution.
    :param preprocessed_data_dir: directory where preprocessed data is located.
    :param describe: if True, writes CSVs of feature descriptions of the
        first-order features.
    """
    # load the data
    preprocessed_data_by_institution = get_data_paths_by_institution(
        preprocessed_data_dir)
    fp = preprocessed_data_by_institution[institution]
    logging.debug("reading data from %s", fp)
    df = pd.read_feather(fp)
    logging.debug("reading data complete; preprocessing data")

    # 'age' is an int for validation, but we cast to float for modeling.
    df["age"] = df["age"].astype(float)

    # Split the data by year
    train_df, test_df, validation_df, eval_df = \
        train_test_validation_eval_split(df)
    # Standardize the data
    train_df, test_df, validation_df, eval_df = scale_data(
        train_df=train_df,
        test_df=test_df,
        validation_df=validation_df,
        eval_df=eval_df)

    train_df = preprocessing.preprocess_fn(train_df)
    test_df = preprocessing.preprocess_fn(test_df)
    validation_df = preprocessing.preprocess_fn(validation_df)
    eval_df = preprocessing.preprocess_fn(eval_df)

    logging.debug("preprocessing complete")

    if describe:
        for splitname, splitdf in zip(
                ("train", "test", "validation", "eval"),
                (train_df, test_df, validation_df, eval_df)):
            write_description(institution, splitname, splitdf,
                              scaled=False)

    return train_df, test_df, validation_df, eval_df


def x_y_split(df, target_column: str = TARGET_COLNAME) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    X = df[[c for c in df.columns if c != target_column]]
    y = df[target_column]
    return X, y


def train_test_validation_eval_split(
        df: pd.DataFrame,
        split_column: str = 'year'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train/test/validation sets.

    The test set is the final year of the data.

    The training, validation, and eval  sets are all previous years of the data;
    of which 10% is allocated to the validation, and 10% to the eval sets,
    with the remainder in the test set. This setup ensures
    that the validation dataset would also be available at training time, so
    it can realistically be used for model selection, and also provides an
    eval set for evaluating model performance on unseen data in-distribution.
    """
    test_year = df[split_column].max()  # save final year for test/validation
    test_df = df.loc[df[split_column] == test_year]
    train_val_df = df.loc[df[split_column] != test_year]
    train_df, eval_validation_df = train_test_split(train_val_df,
                                                    test_size=0.2,
                                                    random_state=RANDOM_STATE)
    validation_df, eval_df = train_test_split(eval_validation_df,
                                              train_size=0.5,
                                              random_state=RANDOM_STATE)
    return train_df, test_df, validation_df, eval_df


def get_data_paths_by_institution(preprocessed_data_dir):
    """
    Fetch a dictionary mapping institution codes to .feather files.

    Each instutitions' data available to an experiment should be stored
    as a .feather file located in
    preprocessed_data_dir with the name `INSTITUTION_CODE`.feather.
    """
    preprocessed_data_by_institution = {
        inst: os.path.join(preprocessed_data_dir, f"{inst}.feather")
        for inst in INSTITUTION_CODES
    }
    return preprocessed_data_by_institution
