from fastapi import FastAPI, HTTPException
import sys
import os
sys.path.append(os.path.abspath("src"))

from src.forecasting.pipeline.prediction_pipeline import PredictionPipeline


app = FastAPI(
    title="Sales Forecast API",
    description="State-wise forecasting using best ML model",
    version="1.0"
)



pipeline = PredictionPipeline()


@app.get("/")
def home():
    return {"message": "Forecast API is running 🚀"}



@app.get("/forecast/{state}")
def forecast_state(state: str):

    try:
        df = pipeline.forecast(state)

        model_used = df["model_used"].iloc[0]

        return {
            "state": state,
            "model_used": model_used,
            "forecast": df[["date","forecast"]].to_dict(orient="records")
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {state}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
