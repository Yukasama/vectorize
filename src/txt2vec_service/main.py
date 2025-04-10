from fastapi import FastAPI
from txt2vec_service.routers import dataset

app = FastAPI(
    title="Text2Vec Service",
    description="Service for text embedding and vector operations",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"message": "Hello, world!"}


app.include_router(dataset.router)
