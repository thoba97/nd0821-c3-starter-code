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
