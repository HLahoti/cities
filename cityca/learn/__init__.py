"""Learning transition rules from observed growth (needs scikit-learn).

* ``features``  — FeatureSet: per-cell predictors from any layers
* ``dataset``   — observed states (events replayed) and transition samples
* ``learners``  — LearnedModel (logistic / gbm), evaluation, permutation importance
* ``train``     — train_from_scenario: one call from a scenario to a saved model
"""

from .dataset import observed_state, observed_years, transitions
from .features import DEFAULT_FEATURES, FeatureSet
from .learners import LEARNERS, LearnedModel, evaluate, permutation_importance
from .train import train_from_scenario

__all__ = ["FeatureSet", "DEFAULT_FEATURES", "observed_state", "observed_years", "transitions",
           "LEARNERS", "LearnedModel", "evaluate", "permutation_importance", "train_from_scenario"]
