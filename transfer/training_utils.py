"""
Utilities for model training, including data splitting etc.
"""
import functools
import logging
import os
from typing import Any, Mapping

import dill as pickle
import lightgbm
import pandas as pd
from scikeras.wrappers import KerasClassifier
import sklearn
import sklearn.linear_model
from sklearn.linear_model import LogisticRegressionCV
import sklearn.neural_network
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

from transfer.experiment_utils import get_model_path, x_y_split, RANDOM_STATE
from transfer import keys, models
from transfer.keys import TARGET_COLNAME

MAX_ITER = 1000

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def _get_model(model_type, d_in: int, **kwargs):
    n_jobs = max(os.cpu_count() - 2, 1)
    class_weight = "balanced"
    if model_type == keys.L2LR:
        if "C" in kwargs:
            # Set penalty to l2 if regularization parameter is nonzero.
            penalty = "l2" if kwargs.get("C") else "none"
            model = sklearn.linear_model.LogisticRegression(
                penalty=penalty,
                class_weight=class_weight,
                n_jobs=n_jobs,
                max_iter=MAX_ITER,
                random_state=RANDOM_STATE,
                **kwargs)
        else:
            # Case: l2lambda is not specified; obtain best value using CV.
            model = LogisticRegressionCV(
                penalty="l2",
                class_weight=class_weight,
                n_jobs=n_jobs,
                max_iter=MAX_ITER,
                random_state=RANDOM_STATE,
                **kwargs
            )
    elif model_type == keys.ADA:
        model = AdaBoostClassifier(
            random_state=RANDOM_STATE,
            **kwargs)
    elif model_type == keys.LIGHTGBM:
        model = lightgbm.LGBMClassifier(class_weight=class_weight, **kwargs)
    elif model_type == keys.MLP:
        model_fn = functools.partial(models.get_keras_model, d_in=d_in,
                                     **kwargs)
        model = KerasClassifier(model=model_fn, verbose=1)

    elif model_type == keys.RF:
        model = RandomForestClassifier(n_jobs=n_jobs,
                                       class_weight=class_weight,
                                       random_state=RANDOM_STATE,
                                       **kwargs)

    else:
        raise NotImplementedError
    return model


def get_tuned_hparams(model):
    """Fetch the tuned model hparams, if model hparams were tuned."""
    if isinstance(model, GridSearchCV):
        return model.best_params_
    elif isinstance(model, LogisticRegressionCV):
        return {"C": model.C_}
    else:
        return None


def _train_model(model, model_type, X_train, y_train):
    """Train a model, handling special case of MLP with GridSearchCV."""
    logger.info("fitting model of type {} with training shape {}.".format(
        type(model), X_train.shape))
    if model_type != keys.MLP:
        model.fit(X_train, y_train)
    else:
        # We cannot pass a dict to GridSearchCV. So, to use class-balanced
        # weights in keras models, we instead use sample_weight.
        class_weight = models.compute_class_weight(y_train)
        sample_weight = y_train.map(class_weight)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def train_model(train_df: pd.DataFrame, validation_df: pd.DataFrame,
                model_type: str,
                target_column: str = TARGET_COLNAME,
                model_kwargs: Mapping[Any, Any] = None,
                hparam_grid=None,
                ) -> Any:
    """Train a model on train_df and evaluate it on eval_df."""
    X_train, y_train = x_y_split(train_df, target_column=target_column)
    X_val, y_val = x_y_split(validation_df, target_column=target_column)

    if model_kwargs is None:
        model_kwargs = {}
    model = _get_model(model_type, d_in=X_train.shape[1], **model_kwargs)
    if hparam_grid:
        model = GridSearchCV(model, hparam_grid,
                             cv=None if model_type != keys.MLP else 3)

    model = _train_model(model, model_type, X_train, y_train)

    return model


def initialize_base_dir(fp: str):
    """If the base directory of fp does not exist, create it."""
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))
    return


def save_model(model, uid):
    """Save a model to disk."""
    if not isinstance(model, tf.keras.Model):
        filepath = get_model_path(uid, ".pickle")
        initialize_base_dir(filepath)
        pickle.dump(model, open(filepath, "wb"))
        # filepath = get_model_path(uid)
        # initialize_base_dir(filepath)
        # dump(model, filepath)
    else:
        # use no suffix, since tf.keras.Model.save() creates a directory
        filepath = get_model_path(uid, suffix="")
        initialize_base_dir(filepath)
        model.save(filepath)
    logging.info("saved model to %s", filepath)
    return


def load_model(uid):
    """
    Load a previously-serialized model.

    :param uid: uid to load.
    :param keras_to_sklearn: if True, cast keras models to a sklearn-compatible
        interface (see models.SkLearnKerasModel). This is required when using
        sklearn ensembling functions that expect classes to implement methods
        .predict_proba().
    :return: the model with trained weights.
    """
    filepath = get_model_path(uid, suffix=".pickle")
    assert os.path.exists(filepath), f"no model file found at {filepath}"
    logging.warning(
        f"loading model from {filepath} "
        f"using sklearn version {sklearn.__version__}"
        "If this is not the same version used to train the models this can lead"
        "to unexpected behavior!")
    try:
        return pickle.load(open(filepath, "rb"))
        # return load(filepath)
    except Exception as e:
        logging.error(f"exception loading model from {filepath}: {e}")
        raise e
