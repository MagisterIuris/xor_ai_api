from pydantic import BaseModel

class PredictionResponse(BaseModel): 
    user_input: list[float]
    raw_output: float 
    prediction: int 