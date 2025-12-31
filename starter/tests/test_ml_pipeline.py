import numpy as np
import pandas as pd
from pathlib import Path

from starter.ml.data import process_data
from starter.ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    load_model,
    save_encoder,
    load_encoder,
)


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


def _sample_dataframe():
    return pd.DataFrame(
        {
            "age": [25, 38, 28],
            "workclass": ["Private", "Self-emp-not-inc", "Private"],
            "education": ["Bachelors", "HS-grad", "Bachelors"],
            "marital-status": ["Never-married", "Married-civ-spouse", "Never-married"],
            "occupation": ["Tech-support", "Craft-repair", "Exec-managerial"],
            "relationship": ["Not-in-family", "Husband", "Not-in-family"],
            "race": ["White", "Black", "Asian-Pac-Islander"],
            "sex": ["Male", "Male", "Female"],
            "native-country": ["United-States", "United-States", "India"],
            "hours-per-week": [40, 50, 30],
            "capital-gain": [0, 0, 0],
            "salary": [">50K", "<=50K", ">50K"],
        }
    )


def test_process_data_returns_expected_types_and_shapes():
    df = _sample_dataframe()
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    # encoder and lb should be fitted sklearn objects with transform methods
    assert hasattr(encoder, "transform")
    assert hasattr(lb, "transform")
    # y should be binary (0/1)
    assert set(np.unique(y)).issubset({0, 1})

    # Now test inference mode using the fitted encoder/lb
    X2, y2, encoder2, lb2 = process_data(df, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    assert isinstance(X2, np.ndarray)
    assert isinstance(y2, np.ndarray)
    assert encoder2 is encoder
    assert lb2 is lb


def test_train_inference_and_metrics_return_correct_types():
    # Small synthetic dataset
    X = np.random.RandomState(0).randint(0, 5, size=(20, 6)).astype(float)
    y = np.random.RandomState(1).randint(0, 2, size=(20,))

    model = train_model(X, y)
    # model must have predict
    assert hasattr(model, "predict")

    preds = inference(model, X[:5])
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 5

    precision, recall, fbeta = compute_model_metrics(y[:5], preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_save_and_load_model_and_encoders(tmp_path: Path):
    df = _sample_dataframe()
    X, y, encoder, lb = process_data(df, categorical_features=cat_features, label="salary", training=True)

    model = train_model(X, y)

    model_path = tmp_path / "model.joblib"
    encoder_path = tmp_path / "encoder.joblib"
    lb_path = tmp_path / "label_binarizer.joblib"

    save_model(model, str(model_path))
    save_encoder(encoder, lb, str(encoder_path), str(lb_path))

    loaded = load_model(str(model_path))
    assert hasattr(loaded, "predict")

    loaded_enc, loaded_lb = load_encoder(str(encoder_path), str(lb_path))
    assert hasattr(loaded_enc, "transform")
    assert hasattr(loaded_lb, "transform")

    # Ensure loaded model can make predictions
    sample = X[:2]
    out = loaded.predict(sample)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 2
