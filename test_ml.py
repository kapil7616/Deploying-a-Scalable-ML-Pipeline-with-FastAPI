import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier


# Load a small subset of the census data for quick testing
@pytest.fixture(scope="module")
def sample_data():
    data = pd.read_csv("data/census.csv")
    # Use only the first 100 rows for faster testing
    return data.head(100)


def test_process_data_returns_expected_shapes(sample_data):
    """
    Test that process_data returns arrays of the expected shape.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )

    # Verify types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    # Check shapes
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] > 0


def test_train_model_returns_random_forest(sample_data):
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_produces_reasonable_values(sample_data):
    """
    Test that compute_model_metrics returns values between 0 and 1.
    """
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        sample_data, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
