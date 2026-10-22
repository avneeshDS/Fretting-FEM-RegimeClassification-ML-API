from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

model = joblib.load(r"V:\VZU\Python\Fretting_models\fretting_fem_regimeclassification_ml_api\fretting_regime_classifier.joblib")

app =FastAPI(title=" Fretting Regime Classification API", 
             description="Gross Slip, Mixed Slip, or Partial Slip under given Input Parameters")

Feature_order = ['P', 'D', 'µ', 'a', 'µP', 'µPD', 'D_over_a']

class FrettingInput(BaseModel):
    P:float = Field(..., gt=0, description="Applied normal load in N", examples=[20.0])
    D:float = Field(..., gt=0, description="Displacement amplitude in microns", examples=[0.025])
    µ:float = Field(..., gt=0, description="CoF", examples=[0.5])
    a:float = Field(..., gt=0, description="Contact radius, calculated from contact mechanics")

class PredictionResponse(BaseModel):
    predicted_regime: str
    derived_features: dict[str, float]

@app.get("/")
def home():
    return {"message": "Fretting API for DE prediction is running", "documentation": "/docs",
            "healthcheck": "/health",}

@app.get("/health")
def healthcheck():
    return {"Status": "Healthy", "model_loaded": "True", "model_name": "model"}

@app.post("/predict")
def predict(data: FrettingInput):
    try:
        µP = data.µ * data.P
        µPD = data.µ * data.P * data.D
        D_over_a = data.D / data.a

        input_dataframe = pd.DataFrame(
            {"P": [data.P],
            "µ": [data.µ],
            "D": [data.D],
            "µP": [µP],
            "µPD": [µPD],
            "a": [data.a],
            "D_over_a": [D_over_a],})
        
        class_names = {0: "Gross", 1: "Mixed", 2: "Partial"}
        input_dataframe = input_dataframe[Feature_order]
        predictions = model.predict(input_dataframe)
        predicted_regime = int(predictions[0])
        resulted_regime = class_names[predicted_regime]

        return PredictionResponse(
            predicted_regime=resulted_regime,
            derived_features={"µP": µP, "µPD": µPD, "D_over_a": D_over_a},)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed{error!s}")


         
         