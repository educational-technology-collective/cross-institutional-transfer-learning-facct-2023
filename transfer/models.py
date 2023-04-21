import logging
from typing import Union

import numpy as np
import pandas as pd

import sklearn
import tensorflow as tf

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def compute_class_weight(y: Union[pd.Series, pd.DataFrame]) -> dict:
    """Compute the class weights for 'balanced' class weighting."""
    classes = np.unique(y.values)
    weights = sklearn.utils.class_weight.compute_class_weight(
        class_weight="balanced", classes=classes, y=y.values)
    return dict(zip(classes, weights))


def keras_predict_proba(model: tf.keras.Model, X):
    preds = model.predict(X)
    return np.column_stack((1 - preds, preds))


def keras_predict(model, X):
    """sklearn-style predict (gives "hard" predictions)"""
    probs = keras_predict_proba(model, X)
    return np.argmax(probs, axis=1)


class SkLearnKerasModel:
    """A limited sklearn-compatible interface for keras models.

    Note that this class only supports prediction (not training or saving),
    as it is intended to be ephemeral. This is because subclasses of keras
    Model objects cannot support several important types of functionality,
    such as proper saving/loading.
    """

    def __init__(self, model: tf.keras.Model):
        self.model = model

    def predict_proba(self, X):
        return keras_predict_proba(self.model, X)

    def predict(self, X):
        return keras_predict(self.model, X)


def get_keras_model(num_layers: int, d_hidden: int, d_in: int,
                    l2lambda: float = 0.):
    model = tf.keras.Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(tf.keras.layers.Dense(
                d_hidden, input_shape=(d_in,), activation='relu',
                kernel_regularizer=tf.keras.regularizers.L2(l2lambda)))
        else:
            model.add(tf.keras.layers.Dense(
                d_hidden, activation='relu',
                kernel_regularizer=tf.keras.regularizers.L2(l2lambda)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def fit_keras_model(model: tf.keras.Model, X, y, X_val=None, y_val=None,
                    sample_weight=None,
                    batch_size=256, epochs=50):
    # use sample weight if provided; otherwise use class weights
    if sample_weight is None:
        class_weight = compute_class_weight(y)
    else:
        class_weight = None

    assert not (sample_weight is not None and class_weight is not None), \
        "sanity check that sample weight and class weight are not both used."

    logging.info(f"fitting keras model")
    validation_data = (X_val, y_val) if X_val is not None else None

    model.fit(X, y, batch_size=batch_size, epochs=epochs,
              class_weight=class_weight,
              validation_data=validation_data,
              sample_weight=sample_weight)
    return model


class StackingClassifier:
    """A stacking classifier.

    Uses the same procedure as sklearn.StackingClassifier with
    passthrough=True, but does *not* train the base estimators.
    """

    def __init__(self, learner, names_and_base_learners):
        self.learner = learner
        self.names_and_base_learners = names_and_base_learners

    @property
    def is_keras(self):
        return isinstance(self.learner, tf.keras.Model)

    def _make_prediction_matrix(self, X: pd.DataFrame):
        voter_preds = list()
        for name, voter in self.names_and_base_learners:
            preds = voter.predict_proba(X)
            assert np.all(~np.isnan(preds)), f"invalid preds for model {name}"

            if len(preds.shape) == 1:
                # Case: single-observation output from keras model; this must
                # be given an extra row dimension: the raw output has shape
                # (num_classes,) and we need shape (1, num_classes).
                preds = np.expand_dims(preds, 0)

            pred_df_cols = [f"{name}_{x}" for x in range(preds.shape[1])]
            pred_df = pd.DataFrame(preds, columns=pred_df_cols)
            voter_preds.append(pred_df)

        voter_pred_df = pd.concat(voter_preds, axis=1)

        assert len(voter_pred_df) == len(X)

        return pd.concat((X.reset_index(drop=True),
                          voter_pred_df.reset_index(drop=True)),
                         axis=1)

    def fit(self, X_tr, y_tr, sample_weight=None):
        X_tr_with_base_preds = self._make_prediction_matrix(X_tr)
        if not self.is_keras:
            self.learner.fit(X_tr_with_base_preds, y_tr,
                             sample_weight=sample_weight)
        else:
            self.learner = fit_keras_model(self.learner, X=X_tr_with_base_preds,
                                           y=y_tr, sample_weight=sample_weight)

    def predict(self, X):
        X_with_base_preds = self._make_prediction_matrix(X)
        if not self.is_keras:
            return self.learner.predict(X_with_base_preds)
        else:
            return keras_predict(self.learner, X_with_base_preds)

    def predict_proba(self, X):
        X_with_base_preds = self._make_prediction_matrix(X)
        if not self.is_keras:
            return self.learner.predict_proba(X_with_base_preds)
        else:
            return keras_predict_proba(self.learner, X_with_base_preds)
