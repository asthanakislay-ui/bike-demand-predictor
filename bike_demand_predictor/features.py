import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from bike_demand_predictor.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CYCLICAL_FEATURES,
    BINARY_FEATURES
)

def encode_hour(X):
    """Encodes hour of day into sine and cosine components to preserve cyclical nature."""
    # Handle both numpy arrays and pandas DataFrames
    if hasattr(X, 'iloc'):
        hour = X.iloc[:, 0].values
    else:
        hour = X[:, 0]

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    return np.column_stack([hour_sin, hour_cos])

def get_preprocessor():
    """
    Returns a scikit-learn ColumnTransformer for preprocessing features.
    Modularizing this allows the same transformation to be used in training and inference.
    """

    numeric_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    cyclical_pipeline = Pipeline(steps=[
        ("hour_encoding", FunctionTransformer(
            encode_hour,
            validate=True
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("cyc", cyclical_pipeline, CYCLICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES)
        ]
    )

    return preprocessor
