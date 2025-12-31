from fastapi.testclient import TestClient
# Import the 'app' from your main API script
from main import app

client = TestClient(app)

def test_get_root():
    """Test the GET method on the root domain."""
    response = client.get("/")
    # Test status code
    assert response.status_code == 200
    # Test contents of the request object
    assert response.json() == {"greeting": "Welcome to the Census Income Prediction API!"}

def test_predict_low_income():
    """Test POST inference for a case likely to result in <=50K."""
    sample = {
        "age": 19,
        "workclass": "Private",
        "fnlgt": 12345,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    # Test that the output matches the first possible model result
    assert response.json()["prediction"] == " <=50K"

def test_predict_high_income():
    """Test POST inference for a case likely to result in >50K."""
    sample = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234567,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    # Test that the output matches the second possible model result
    assert response.json()["prediction"] == " >50K"