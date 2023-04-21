"""
Script to train a model on a given instutitions' dataset.

Usage:

python scripts/train.py --src_institution my_institution --lambda_sweep
"""
import argparse
import logging
from typing import List, Mapping, Any

from transfer import training_utils, experiment_utils
from transfer.experiment_utils import DEFAULT_MODEL_KWARGS, HPARAM_GRID
from transfer.keys import INSTITUTION_CODES, MLP, L2LR, LIGHTGBM
from transfer.metrics import postprocess_metrics, concatenate_and_write_metrics, \
    metrics_src_filepath
from transfer.evaluation import fetch_subgroup_metrics, fetch_prediction_metrics

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

# Maps model names to the name of the l2 regularization parameter for that
# model's constructor.
MODEL_l2LAMBDA_PARAM_NAMES = {
    MLP: 'l2lambda',
    LIGHTGBM: 'reg_lambda',
    L2LR: 'C',  # note: this is *inverse* regularization strength
}


def _update_l2lambda_param(model_kwargs: Mapping[str, Any],
                           model_type: str,
                           l2lambda_val: float) -> Mapping[str, Any]:
    l2lambda_param = MODEL_l2LAMBDA_PARAM_NAMES[model_type]
    if model_type != L2LR:
        model_kwargs.update({l2lambda_param: l2lambda_val})
    else:
        # For sklearn LogisticRegression, it takes the inverse of the
        # l2 regularization strength as a parameter.
        model_kwargs.update(
            {l2lambda_param: 0. if l2lambda_val == 0. else 1. / l2lambda_val})
    return model_kwargs


def compute_metrics(model, df, uid, split, src_institution):
    metrics = fetch_subgroup_metrics(model, df)
    metrics[f"full_{split}"] = fetch_prediction_metrics(
        model, *training_utils.x_y_split(df))

    tuned_hparams = training_utils.get_tuned_hparams(model)
    metrics = postprocess_metrics(
        metrics, uid, src_institution, src_institution, split=split,
        tuned_hparams=str(tuned_hparams))
    return metrics


def main(src_institution: str, data_dir: str, model_types: List,
         lambda_sweep: bool = False, interaction_degree=1,
         tune_hparams=True):
    logger.info("src institution is %s", src_institution)
    metrics_dfs = list()
    # load the data
    train_df, test_df, validation_df, eval_df = experiment_utils.read_and_split_data(
        src_institution, preprocessed_data_dir=data_dir)

    # Train the base models.
    for model_type in model_types:
        uid = experiment_utils.make_uid(
            src_institution=src_institution,
            model_type=model_type,
            interaction_degree=interaction_degree)

        # train and save the model
        logging.info("training model with uid %s", uid)
        model = training_utils.train_model(
            train_df, validation_df, model_type,
            model_kwargs=DEFAULT_MODEL_KWARGS[model_type],
            hparam_grid=None if not tune_hparams else HPARAM_GRID[model_type])

        # Compute test metrics
        test_metrics = compute_metrics(
            model, test_df, uid, src_institution=src_institution, split="test")
        metrics_dfs.append(test_metrics)

        # Compute eval metrics
        eval_metrics = compute_metrics(
            model, eval_df, uid, src_institution=src_institution, split="eval")
        metrics_dfs.append(eval_metrics)

        training_utils.save_model(model, uid)

    # Sweep over L2 regularization values for each model.
    if lambda_sweep:
        for model_type in model_types:
            for lambda_value in experiment_utils.LAMBDA_GRID:
                uid = experiment_utils.make_uid(
                    src_institution=src_institution,
                    interaction_degree=interaction_degree,
                    model_type=model_type,
                    l2lambda=lambda_value)

                # train and save the model
                model_kwargs = DEFAULT_MODEL_KWARGS[model_type]
                model_kwargs = _update_l2lambda_param(model_kwargs, model_type,
                                                      lambda_value)
                logging.info("training model with uid %s", uid)
                model = training_utils.train_model(
                    train_df, validation_df, model_type=model_type,
                    model_kwargs=model_kwargs,
                    hparam_grid=None if not tune_hparams else HPARAM_GRID[
                        model_type])

                # Compute test metrics
                test_metrics = compute_metrics(
                    model, test_df, uid, src_institution=src_institution,
                    split="test")
                metrics_dfs.append(test_metrics)

                # Compute eval metrics
                eval_metrics = compute_metrics(
                    model, eval_df, uid, src_institution=src_institution,
                    split="eval")
                metrics_dfs.append(eval_metrics)

                training_utils.save_model(model, uid)

    src_filename = metrics_src_filepath(src_institution)
    concatenate_and_write_metrics(metrics_dfs, src_filename)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_institution", choices=INSTITUTION_CODES,
        required=True,
        help="abbreviation for the source/training data institution.")
    parser.add_argument(
        "--model_types",
        default=[L2LR, LIGHTGBM, MLP],
        help="model types to use; can be specified multiple times.",
        nargs="+", action="append")
    parser.add_argument(
        "--data_dir", default="./data/preprocessed",
        help="path to a directory containing the preprocessed data.")
    parser.add_argument("--lambda_sweep", default=False, action="store_true",
                        help="whether to run sweep of models over l2 lambda"
                             "values; if false, only trains the base models.")
    args = parser.parse_args()
    main(**vars(args))
