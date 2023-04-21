"""
Script to evaluate a set of trained models on a given institution's dataset.

Will attempt to evaluate models from all possible source institutions,
of all possible model types, on the given target institution. This includes
a sweep of L2LR regularization values.

Usage:

python scripts/evaluate.py --target_institution my_institution --lambda_sweep
"""
import argparse
import logging
from typing import List
import pandas as pd
import transfer.evaluation
from transfer.evaluation import fetch_subgroup_metrics
import transfer.experiment_utils
from transfer.metrics import postprocess_metrics, concatenate_and_write_metrics
from transfer import training_utils, experiment_utils, keys

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def evaluate_model_from_uid(df: pd.DataFrame, uid: str, src_institution: str,
                            target_institution: str, split: str):
    """Helper function to run evaluation for a fixed uid."""
    try:
        model = training_utils.load_model(uid)
    except Exception as e:
        logging.warning("skipping model uid %s: %s", uid, e)
        return

    logger.info("src institution is %s, target institution is %s",
                src_institution, target_institution)
    logger.info("experiment uid is %s", uid)

    metrics = transfer.evaluation.fetch_prediction_metrics(
        model, *transfer.experiment_utils.x_y_split(df))
    subgroup_metrics = fetch_subgroup_metrics(model, df)
    subgroup_metrics[f"full_{split}"] = metrics

    metrics_df = postprocess_metrics(
        subgroup_metrics, uid,
        src_institution=src_institution,
        target_institution=target_institution,
        split=split)
    return metrics_df


def main(target_institution: str, model_types: List, data_dir: str,
         interaction_degree=1, lambda_sweep: bool = False):
    _, test_df_tgt, _, eval_df_tgt = experiment_utils.read_and_split_data(
        target_institution, preprocessed_data_dir=data_dir)

    metrics_dfs = list()

    for src_institution in keys.INSTITUTION_CODES:

        # Evaluate the base models
        for model_type in model_types:
            # load the model
            uid = experiment_utils.make_uid(
                src_institution=src_institution,
                model_type=model_type,
                interaction_degree=interaction_degree)

            test_metrics_df = evaluate_model_from_uid(test_df_tgt, uid,
                                                      src_institution,
                                                      target_institution,
                                                      split="test")
            metrics_dfs.append(test_metrics_df)

            eval_metrics_df = evaluate_model_from_uid(eval_df_tgt, uid,
                                                      src_institution,
                                                      target_institution,
                                                      split="eval")
            metrics_dfs.append(eval_metrics_df)

            if lambda_sweep:
                for lambda_value in experiment_utils.LAMBDA_GRID:
                    uid = experiment_utils.make_uid(
                        src_institution=src_institution,
                        model_type=model_type,
                        interaction_degree=interaction_degree,
                        l2lambda=lambda_value)
                    test_metrics_df = evaluate_model_from_uid(
                        test_df_tgt, uid,
                        src_institution,
                        target_institution,
                        split="test")
                    metrics_dfs.append(test_metrics_df)

                    eval_metrics_df = evaluate_model_from_uid(
                        eval_df_tgt, uid,
                        src_institution,
                        target_institution,
                        split="eval")
                    metrics_dfs.append(eval_metrics_df)

    concatenate_and_write_metrics(
        metrics_dfs, f"metrics_tgt_{target_institution}.csv")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_institution", choices=keys.INSTITUTION_CODES,
        required=True,
        help="abbreviation for the target/test data institution.")
    parser.add_argument(
        "--data_dir", default="./data/preprocessed",
        help="path to a directory containing the preprocessed data.")
    parser.add_argument(
        "--model_types",
        default=[
            keys.L2LR,
            keys.LIGHTGBM,
            keys.MLP
        ],
        help="model types to use; can be specified multiple times.",
        nargs="+", action="append")
    parser.add_argument("--lambda_sweep", default=False, action="store_true",
                        help="whether to run sweep of models over l2 lambda"
                             "values; if false, only trains the base models.")
    args = parser.parse_args()
    main(**vars(args))
