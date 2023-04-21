import logging
import os
import time
from typing import List
from transfer.keys import METRICS_DIR

import pandas as pd

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def postprocess_metrics(metrics: dict, uid: str, src_institution: str,
                        target_institution: str,
                        split: str, print_summary=True,
                        **kwargs) -> pd.DataFrame:
    """Postprocess metrics, adding relevant metadata columns."""
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(
        columns={"index": "subgroup"})
    metrics_df["src_institution"] = src_institution
    metrics_df["target_institution"] = target_institution
    metrics_df["split"] = split

    # postprocess the metrics.
    metrics_df["uid"] = uid
    metrics_df["timestamp"] = time.time()

    if kwargs:
        for k, v in kwargs.items():
            metrics_df[k] = v

    if print_summary:
        logging.info(
            "printing {} metrics with source "
            "institution {} on target institution {}:".format(
                split, src_institution, target_institution))
        print_basic_metrics(metrics_df)
    return metrics_df


def write_metrics(df, filepath):
    df.to_csv(filepath, index=False)
    logging.info("metrics saved to %s", filepath)


def metrics_tgt_filename(target_institution: str, dirname=METRICS_DIR) -> str:
    fname = f"metrics_tgt_{target_institution}_ensembles.csv"
    return os.path.join(dirname, fname)


def metrics_src_filepath(src_institution: str, dirname=METRICS_DIR) -> str:
    fname = f"metrics_src_{src_institution}.csv"
    return os.path.join(dirname, fname)


def concatenate_and_write_metrics(metrics_dfs: List, filename):
    df = pd.concat(metrics_dfs)
    write_metrics(df, filename)


def print_basic_metrics(metrics_df):
    print(metrics_df[
              ["subgroup", "auc", "accuracy", "avg_precision_score",
               "n_test"]])
    return
