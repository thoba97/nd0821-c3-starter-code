# Script to train machine learning model.

import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Make sure package imports work when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    save_model,
    save_encoder,
    slice_performance,
)


def main() -> None:
    data_path = Path(__file__).resolve().parents[1] / "data" / "census.csv"
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Train / test split
    train, test = train_test_split(df, test_size=0.20, random_state=42)

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

    # Process training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {fbeta:.4f}")

    # Save model and encoders
    model_dir = Path(__file__).resolve().parents[1] / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    encoder_path = model_dir / "encoder.joblib"
    lb_path = model_dir / "label_binarizer.joblib"

    save_model(model, str(model_path))
    save_encoder(encoder, lb, str(encoder_path), str(lb_path))

    print(f"Saved model to: {model_path}")
    print(f"Saved encoder to: {encoder_path}")
    print(f"Saved label binarizer to: {lb_path}")

    # Compute slice performance for a categorical feature and save to file
    slice_file = model_dir / "slice_output.txt"
    print(f"Computing slice performance for 'education' and writing to {slice_file}")
    slice_performance(
        model=model,
        df=df,
        feature="education",
        categorical_features=cat_features,
        label="salary",
        encoder=encoder,
        lb=lb,
        out_file=str(slice_file),
    )
    print(f"Slice performance written to: {slice_file}")


if __name__ == "__main__":
    main()
