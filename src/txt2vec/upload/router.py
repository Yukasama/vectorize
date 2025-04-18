from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from txt2vec_service.model_service import load_model_with_tag

model_router = APIRouter(tags=["Model Upload"])

class LoadModelRequest(BaseModel):
    model_id: str
    tag: str

@model_router.post("/load")
def load_model(request: LoadModelRequest):
    try:
        load_model_with_tag(request.model_id, request.tag)
        return {"message": f"Model '{request.model_id}' with tag '{request.tag}' loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

