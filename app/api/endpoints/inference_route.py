from fastapi import Depends
from app.dependencies import get_inference_service
from app.api import router 
from app.schemas import InputData, PredictionResponse
from app.services import InferenceService

@router.post("/predict")
def predict(data: InputData, inference_service: InferenceService = Depends(get_inference_service)):
    user_input, raw_output, prediction =  inference_service.infer(data)
    return PredictionResponse(user_input=user_input, raw_output=raw_output, prediction=prediction)


