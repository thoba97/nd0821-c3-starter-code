from typing import Optional
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path: str):
    """Save trained model to disk using joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str):
    """Load a model saved with joblib."""
    return joblib.load(path)


def save_encoder(encoder, lb, encoder_path: str, lb_path: str):
    """Save encoder and label binarizer to disk."""
    Path(encoder_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)


def load_encoder(encoder_path: str, lb_path: str):
    """Load encoder and label binarizer from disk."""
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return encoder, lb


def slice_performance(
    model,
    df,
    feature: str,
    categorical_features: list,
    label: str,
    encoder,
    lb,
    out_file: Optional[str] = None,
):
    """Compute model metrics for each unique value (slice) of a categorical feature.

    For each unique value in `df[feature]`, the function filters the dataframe,
    processes the data with the provided `encoder` and `lb`, runs the model
    predictions, computes precision/recall/f1, and collects the results.

    If `out_file` is provided, the human-readable results are written there.

    Returns a dict mapping value -> (precision, recall, f1).
    """
    from starter.ml.data import process_data

    results = {}
    lines = []

    # Normalize column names to help avoid mismatches (strip whitespace)
    df = df.copy()
    df.columns = df.columns.str.strip()

    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame columns: {list(df.columns)}")

    values = sorted(df[feature].dropna().unique())

    for val in values:
        df_slice = df[df[feature] == val]
        if df_slice.shape[0] == 0:
            continue

        X_slice, y_slice, _, _ = process_data(
            df_slice,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )

        if X_slice.shape[0] == 0:
            continue

        preds = model.predict(X_slice)
        precision, recall, f1 = compute_model_metrics(y_slice, preds)
        results[val] = (precision, recall, f1)

        lines.append(f"{feature}={val} => precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")

    if out_file is not None:
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    return results