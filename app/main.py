from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app import model_loader
from app.schemas import PredictionRequest, PredictionResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_loader.load_artifacts()
    yield


app = FastAPI(
    title='House Price Prediction API',
    description='Predicting the price of an apartment based on a listing from India',
    version='1.0.0',
    lifespan=lifespan,
)

@app.get('/health', tags=['Service'])
def heatlh():
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {'status': 'healthy'}

@app.post('/predict', response_model=PredictionResponse, tags=['Prediction'])
def predict(request: PredictionRequest):
    if model_loader.model is None or model_loader.target_encoder is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    input_df = pd.DataFrame([request.model_dump()])

    input_df = model_loader.target_encoder.transform(input_df)

    y_pred_log = model_loader.model.predict(input_df)

    y_pred = float(np.expm1(y_pred_log)[0])

    if y_pred >= 10_000_000:
        formatted = f'{y_pred / 10_000_000:.2f} Cr'
    elif y_pred >= 100_000:
        formatted = f'{y_pred / 100_000:.2f} Lac'
    else:
        formatted = f'{y_pred:,.0f} rupees'

    return PredictionResponse(
        predicted_price=y_pred,
        predicted_price_formatted=formatted,
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)