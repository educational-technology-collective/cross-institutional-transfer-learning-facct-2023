from dataclasses import dataclass
import logging
from typing import List, Any, Tuple, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from transfer.training_utils import _get_model, load_model, _train_model
from transfer.metrics import metrics_src_filepath
from transfer.models import StackingClassifier
from transfer.experiment_utils import x_y_split
from transfer import keys

MLP_ONLY_RULE = "mlp_only"
LIGHTGBM_ONLY_RULE = "gbm_only"
L2LR_ONLY_RULE = "l2lr_only"
KITCHEN_SINK_RULE = "kitchen_sink"

BEST_FAIRNESS_RULE = "best_fairness"
BEST_AUC_RULE = "best_auc"

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

ENS_SELECTION_RULES = (BEST_AUC_RULE, BEST_FAIRNESS_RULE, KITCHEN_SINK_RULE,
                       L2LR_ONLY_RULE, LIGHTGBM_ONLY_RULE, MLP_ONLY_RULE)


@dataclass
class NamedVoter:
    name: str
    voter: Any


def best_metric_rule(df: pd.DataFrame, metric: str,
                     higher_is_better: bool,
                     subgroup: str = "full_eval") -> str:
    """Returns UID of model with best metric value on the specified subgroup."""
    assert metric in df.columns
    if higher_is_better:
        idx = df.query(f"subgroup == '{subgroup}'")[metric].idxmax()
    else:
        idx = df.query(f"subgroup == '{subgroup}'")[metric].idxmin()
    return df.iloc[idx]['uid']


def best_fairness_rule(df: pd.DataFrame, metric: str = "auc",
                       split: str = "eval", higher_is_better=True) -> str:
    """Returns UID of model with lowest metric disparity, over all subgroups."""
    split_df = df.query(f"split == '{split}'")
    overall_df = split_df.query(f"subgroup == 'full_{split}'")[['uid', metric]]
    tmp = split_df.groupby('uid') \
        .apply(lambda x: abs(x[metric].max() - x[metric].min())) \
        .rename("disparity") \
        .reset_index('uid')
    tmp = tmp.merge(overall_df, on='uid')
    best_uid = tmp.sort_values(
        by=['disparity', metric],
        ascending=[True, not higher_is_better]).iloc[0, :]["uid"]
    return best_uid


def same_family_rule(df, model_type: str) -> Sequence[str]:
    """Returns all model UIDs of model_type."""
    rows = df[df['uid'].str.contains(f"model_type{model_type}")]
    return rows['uid'].unique().tolist()


def kitchen_sink_rule(df) -> Sequence[str]:
    """Returns all models."""
    return df['uid'].unique().tolist()


def apply_selection_rule(selection_rule, src_institution
                         ) -> Sequence[NamedVoter]:
    """Apply a selection rule according to performance at src_institution."""
    metrics = pd.read_csv(metrics_src_filepath(src_institution))

    if selection_rule == BEST_AUC_RULE:
        uids = [best_metric_rule(metrics, metric="auc", higher_is_better=True)]

    elif selection_rule == BEST_FAIRNESS_RULE:
        uids = [best_fairness_rule(metrics)]

    elif selection_rule == MLP_ONLY_RULE:
        uids = same_family_rule(metrics, model_type=keys.MLP)

    elif selection_rule == LIGHTGBM_ONLY_RULE:
        uids = same_family_rule(metrics, model_type=keys.LIGHTGBM)

    elif selection_rule == L2LR_ONLY_RULE:
        uids = same_family_rule(metrics, model_type=keys.L2LR)

    elif selection_rule == KITCHEN_SINK_RULE:
        uids = kitchen_sink_rule(metrics)

    else:
        raise NotImplementedError

    selected = tuple(NamedVoter(name=uid, voter=load_model(uid))
                     for uid in uids)
    return selected


def get_voters_by_selection_rule(
        selection_rule: str,
        src_institutions: Sequence[str]) -> Sequence[NamedVoter]:
    """Apply a selection rule at src_institutions to fetch NamedVoters."""
    named_voters = list()
    assert selection_rule in ENS_SELECTION_RULES
    for src_institution in src_institutions:
        voters = apply_selection_rule(selection_rule, src_institution)
        named_voters.extend(voters)
    return named_voters


def build_voting_ensemble(voters: Sequence[NamedVoter],
                          y: np.ndarray):
    """Build a majority-voting classifier from a set of model filepaths."""
    names_and_voters = [(x.name, x.voter) for x in voters]
    ensemble = VotingClassifier(names_and_voters,
                                voting="soft")

    # Workaround to ensure VotingClassifier works without needing to .fit();
    # see https://stackoverflow.com/a/54610569/5843188
    voters = [x[1] for x in names_and_voters]
    ensemble.estimators_ = voters

    ensemble.le_ = LabelEncoder().fit(y)
    ensemble.classes_ = ensemble.le_.classes_

    return ensemble


def build_and_train_stacked_ensemble(
        voters: Sequence[NamedVoter], df_train,
        model_type: str, model_kwargs=None, hparam_grid=None, num_classes=2):
    """Create a stacked ensemble, and train it.

    Note that we do not use the sklearn.StackingClassifier task because it
    will re-fit the base models on the target data.
    """
    names_and_voters = [(x.name, x.voter) for x in voters]
    X_tr, y_tr = x_y_split(df_train)

    if model_kwargs is None:
        model_kwargs = {}

    d_in = X_tr.shape[1] + num_classes * len(names_and_voters)
    model = _get_model(model_type, d_in=d_in, **model_kwargs)
    if hparam_grid:
        model = GridSearchCV(model, hparam_grid,
                             cv=None if model_type != keys.MLP else 3)

    ensemble = StackingClassifier(model, names_and_voters)
    logging.info("training stacked ensemble model with type %s.", model_type)
    ensemble = _train_model(ensemble, model_type, X_tr, y_tr)
    return ensemble
