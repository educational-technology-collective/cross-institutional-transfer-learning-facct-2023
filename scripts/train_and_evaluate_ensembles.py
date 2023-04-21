"""
Script to create a majority-voting ensemble from a set of pretrained
models and evaluate on a target institution.

The pretrained models can be from any institution and an arbitrary number
of models can be provided. The filepaths to the trained models must be
provided explicitly.

By default, this script only uses logistic regression models, although
additional model types can be speficified via the model_types flag.

Usage:
python scripts/train_and_evaluate_ensembles.py --target_institution um
"""

import argparse
from itertools import product
import logging
from typing import List, Sequence

import pandas as pd

import transfer.evaluation
from transfer import training_utils, experiment_utils, keys
from transfer.ensembles import build_voting_ensemble, \
    build_and_train_stacked_ensemble, get_voters_by_selection_rule, \
    ENS_SELECTION_RULES, NamedVoter
from transfer.metrics import postprocess_metrics, write_metrics, \
    metrics_tgt_filename
from transfer.evaluation import fetch_subgroup_metrics
from transfer.experiment_utils import HPARAM_GRID

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def evaluate_majority_voting_ensemble(
        target_institution: str,
        src_institutions: Sequence[str],
        voters: Sequence[NamedVoter],
        data_dir: str,
        selection_rule: str,
) -> pd.DataFrame:
    _, test_df, _, _ = experiment_utils.read_and_split_data(
        target_institution, preprocessed_data_dir=data_dir)
    uid = experiment_utils.make_uid(target_institution=target_institution,
                                    model_type=keys.MVE)
    logging.info("building voting ensemble for uid %s", uid)
    model = build_voting_ensemble(voters,
                                  y=test_df[keys.TARGET_COLNAME].values)

    # Compute metrics.
    test_metrics = fetch_subgroup_metrics(model, test_df)
    test_metrics["full_test"] = transfer.evaluation.fetch_prediction_metrics(
        model, *experiment_utils.x_y_split(test_df))
    logging.info(
        "majority-voting ensemble test metrics on target institution:")
    test_metrics = postprocess_metrics(
        test_metrics, uid=uid,
        src_institution='-'.join(src_institutions),
        target_institution=target_institution, split="test")

    test_metrics["selection_rule"] = selection_rule
    test_metrics["ensemble_type"] = keys.MVE
    test_metrics["stacked_model_type"] = None
    return test_metrics


def train_and_evaluate_stacked_ensemble(
        target_institution: str,
        voters: Sequence[NamedVoter],
        data_dir: str,
        selection_rule: str,
        stacked_model_type: str,
        tune_hparams=True
) -> pd.DataFrame:
    train_df, test_df, val_df, _ = experiment_utils.read_and_split_data(
        target_institution, preprocessed_data_dir=data_dir)
    uid = experiment_utils.make_uid(target_institution=target_institution,
                                    model_type=keys.MVE)
    logging.info(
        "building stacked ensemble for uid {} rule {} stacked {}".format(
            uid, selection_rule, stacked_model_type))

    hparam_grid = None if not tune_hparams else HPARAM_GRID[stacked_model_type]
    model = build_and_train_stacked_ensemble(
        voters,
        df_train=train_df,
        model_type=stacked_model_type, hparam_grid=hparam_grid)

    # Compute metrics.
    test_metrics = fetch_subgroup_metrics(model, test_df)
    test_metrics["full_test"] = transfer.evaluation.fetch_prediction_metrics(
        model, *experiment_utils.x_y_split(test_df))
    test_metrics = postprocess_metrics(
        test_metrics, uid=uid,
        src_institution='-'.join(keys.INSTITUTION_CODES),
        target_institution=target_institution, split="test")
    test_metrics["selection_rule"] = selection_rule
    test_metrics["ensemble_type"] = keys.SE
    test_metrics["stacked_model_type"] = stacked_model_type
    return test_metrics


def main(target_institution: str,
         data_dir: str,
         stacked_model_types: str):
    metrics_dfs = []
    src_institutions = [i for i in keys.INSTITUTION_CODES if
                        i != target_institution]
    for selection_rule in ENS_SELECTION_RULES:

        # Majority-voting ensembles
        voters = get_voters_by_selection_rule(selection_rule, src_institutions)
        df = evaluate_majority_voting_ensemble(target_institution,
                                               src_institutions,
                                               voters,
                                               data_dir=data_dir,
                                               selection_rule=selection_rule)
        metrics_dfs.append(df)

        # Stacked ensembles
        for stacked_model_type in stacked_model_types:
            df = train_and_evaluate_stacked_ensemble(
                target_institution,
                voters=voters,
                selection_rule=selection_rule,
                stacked_model_type=stacked_model_type,
                data_dir=data_dir)
            metrics_dfs.append(df)

    # Write the metrics.
    metrics_df = pd.concat(metrics_dfs)
    metrics_filename = metrics_tgt_filename(target_institution)
    write_metrics(metrics_df, metrics_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_institution", choices=keys.INSTITUTION_CODES,
        default=None,
        required=True,
        help="abbreviation for the testing data institution."
             " Defaults to the training institution.")
    parser.add_argument(
        "--data_dir", default="./data/preprocessed",
        help="path to a directory containing the preprocessed data.")
    parser.add_argument(
        "--stacked_model_types", default=[
            keys.L2LR,
            keys.LIGHTGBM,
            keys.MLP
        ],
        nargs="+", action="append",
        help="model types to use for the stacked model a.k.a.'meta-learner'.")
    args = parser.parse_args()
    main(**vars(args))
