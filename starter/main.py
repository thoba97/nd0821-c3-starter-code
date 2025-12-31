import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from ml.data import process_data
from ml.model import load_model, load_encoder, inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Initialize the app
app = FastAPI(
    title="Census Income Prediction API",
    description="An API to predict if an individual's income exceeds $50k.",
    version="1.0.0"
)

# Define paths to saved artifacts
base_path = Path(__file__).resolve().parent
model_path = base_path / "model" / "model.joblib"
encoder_path = base_path / "model" / "encoder.joblib"
lb_path = base_path / "model" / "label_binarizer.joblib"

# Load model and encoders at startup
model = load_model(model_path)
encoder, lb = load_encoder(encoder_path, lb_path)

# Categorical features as defined in your training script
cat_features = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country"
]

# Pydantic model for input data validation and auto-documentation
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., alias="occupation", example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


@app.get("/")
async def welcome():
    """Root GET endpoint with a greeting."""
    return {"greeting": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    """POST endpoint for model inference."""
    # Convert Pydantic model to DataFrame
    # Use by_alias=True to ensure keys match CSV column names (e.g., 'education-num')
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])

    # Process data using your existing logic
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    preds = inference(model, X)
    
    # Use the label binarizer to convert numeric prediction back to string (e.g., >50K)
    prediction = lb.inverse_transform(preds)[0]

    return {"prediction": str(prediction)}