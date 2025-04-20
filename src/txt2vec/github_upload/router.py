from fastapi import APIRouter, HTTPException
from txt2vec.model_registry.service import handle_model_download
from txt2vec.model_registry.schemas import ModelRequest

router = APIRouter()


@router.post("/add_model")
async def add_model(request: ModelRequest):
    try:
        return await handle_model_download(request.github_url)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
