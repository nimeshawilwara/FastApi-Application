from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import List

# Load your trained model
model = joblib.load('california_housing_model.pkl')  # Ensure the model filename matches what you saved earlier

# Define the input data schema
class InputData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

app = FastAPI()

@app.post("/predict")
def predict(data: List[InputData]):
    # Convert input data to the model's expected format
    input_features = [[
        item.MedInc,
        item.HouseAge,
        item.AveRooms,
        item.AveBedrms,
        item.Population,
        item.AveOccup,
        item.Latitude,
        item.Longitude
    ] for item in data]
    
    # Make predictions using the model
    prediction = model.predict(input_features)
    return {"predictions": prediction.tolist()}

