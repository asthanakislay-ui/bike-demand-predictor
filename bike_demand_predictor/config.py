from pathlib import Path

# Feature Groups
NUMERIC_FEATURES = ["temp", "hum", "windspeed"]
CATEGORICAL_FEATURES = ["season", "mnth", "weekday", "weathersit"]
CYCLICAL_FEATURES = ["hr"]
BINARY_FEATURES = ["yr", "holiday", "workingday"]

TARGET_COLUMN = "cnt"
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + CYCLICAL_FEATURES + BINARY_FEATURES

# Model Hyperparameters
DEFAULT_LR = 0.001
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 100
HIDDEN_LAYERS = [64, 32]

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
