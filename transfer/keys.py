"""Shared constants."""

# Model types
L2LR = "l2lr"
RF = "random_forest"
ADA = "adaboost"
LIGHTGBM = "lightgbm"
MLP = "mlp"
MVE = "majority_voting_ensemble"
SE = "stacked_ensemble"

# The interaction degrees to search over; note that higher-order features
# can be extremely memory-intensive as they are quadratic in the input dimension
INTERACTION_DEGREES = [1, ]

# Model kwarg names
L2LAMBDA = "l2lambda"

ENSEMBLE = "ensemble"

MODEL_DIR = "./models"
METRICS_DIR = "./metrics"
TARGET_COLNAME = "retention"

# Data coding keys
MISSING_VALUE = "MISSING"  # use this for categorical data with missing value
