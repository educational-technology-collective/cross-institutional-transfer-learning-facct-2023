from collections import defaultdict
import itertools
import logging
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from transfer.preprocessing import make_dummy_colname
from transfer.experiment_utils import x_y_split
from transfer.validation import CATEGORICAL_FEATURE_ALLOWED_VALS

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def fetch_prediction_metrics(model, X, y) -> dict:
    """Evaluate the model and return a dict mapping metric names to values."""

    y_hat_probs = model.predict_proba(X)
    y_hat_labels = model.predict(X)

    accuracy = sklearn.metrics.accuracy_score(y, y_hat_labels)

    if len(np.unique(y)) > 1:
        auc = sklearn.metrics.roc_auc_score(y, y_hat_probs[:, 1])
        cross_entropy = sklearn.metrics.log_loss(y, y_hat_probs)
        avg_precision = sklearn.metrics.average_precision_score(
            y, y_hat_probs[:, 1])

    else:
        # Case: Only one class present in y_true; these metrics are undefined.
        auc = np.nan
        cross_entropy = np.nan
        avg_precision = np.nan

    f1_score = sklearn.metrics.f1_score(y, y_hat_labels)
    precision = sklearn.metrics.precision_score(y, y_hat_labels)
    recall = sklearn.metrics.recall_score(y, y_hat_labels)

    metrics = {
        "accuracy": accuracy,
        "auc": auc,
        "avg_precision_score": avg_precision,
        "f1": f1_score,
        "precision": precision,
        "recall": recall,
        "cross_entropy": cross_entropy,
        "n_test": len(y)
    }

    for i, j in itertools.product([0, 1], [0, 1]):
        metrics[f"cm_true{i}_pred{j}"] = np.logical_and(
            y == i, y_hat_labels == j).sum()

    return metrics


def compute_metrics_on_subset(full_df: pd.DataFrame, model, idxs: pd.Series):
    if not np.any(idxs.values):
        return None

    df = full_df.loc[idxs]
    X, y = x_y_split(df)
    metrics = fetch_prediction_metrics(model, X, y)
    return metrics


def fetch_subgroup_metrics(
        model, df: pd.DataFrame,
        sensitive_subgroups=("sex", "urm_status"),
        min_subgroup_size: int = 2) -> dict:
    """Creates a nested dictionary with subgroup names as top-level keys.

    For each subgroup, the value is a dictionary of metric_name:metric_value
    mappings.
    """
    metrics = defaultdict(dict)
    # Build a list of the unique values in each group
    g0_key = sensitive_subgroups[0]
    g0_values = CATEGORICAL_FEATURE_ALLOWED_VALS[g0_key]
    g1_key = sensitive_subgroups[1]
    g1_values = CATEGORICAL_FEATURE_ALLOWED_VALS[g1_key]
    intersections = [(i, j) for i, j in
                     itertools.product(g0_values, g1_values)]
    # TODO(jpgard): implement this using fairlearn.MetricFrame instead.
    # Intersectional metrics
    for g0_val, g1_val in intersections:
        # fetch the subset of the data containing
        # the intersection of these identities
        g0_colname = make_dummy_colname(g0_key, g0_val)
        g1_colname = make_dummy_colname(g1_key, g1_val)
        intersection_idxs = (df[g0_colname] == 1) & (df[g1_colname] == 1)
        if intersection_idxs.sum() >= min_subgroup_size:
            logger.debug(f"computing subgroup metrics for {g0_colname}, "
                         f"{g1_colname} with model of type {type(model)}")
            # Case: sklearn.ensemble._voting.predict() fails on size-1 datasets
            # with a keras predictor that has only single-dimension output;
            # we skip subgroup datasets of size 1.
            subgroup_metrics = compute_metrics_on_subset(
                df, model, idxs=intersection_idxs)
        else:
            logger.debug(f"skipping computing subgroup metrics for {g0_colname}, "
                         f"{g1_colname} with model of type {type(model)} due "
                         f"to insufficient subgroup size (<= "
                         f"{min_subgroup_size})")
            subgroup_metrics = {}

        subgroup_str = "{}{}_{}{}".format(
            g0_key, g0_val, g1_key, g1_val)
        metrics[subgroup_str] = subgroup_metrics

    # Marginal metrics for sensitive subgroups
    marginal_cols_and_values = \
        [(g0_key, val) for val in g0_values] + \
        [(g1_key, val) for val in g1_values]

    for key, val in marginal_cols_and_values:
        # fetch the subset of the data containing
        # the intersection of these identities
        colname = make_dummy_colname(key, val)
        intersection_idxs = df[colname] == 1
        subgroup_metrics = compute_metrics_on_subset(
            df, model, idxs=intersection_idxs)
        subgroup_str = "{}{}".format(key, val)
        metrics[subgroup_str] = subgroup_metrics

    return metrics
