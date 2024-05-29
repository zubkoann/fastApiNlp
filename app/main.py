from fastapi import FastAPI
import nltk
from contextlib import asynccontextmanager
from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.api.endpoints import router
from app.api.doc2vec import doc2vec_router

from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title=settings.PROJECT_NAME)
training_in_progress = False
model_trained = False


app.include_router(router)
app.include_router(doc2vec_router)

app.include_router(heartbeat_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH))
app.add_event_handler("shutdown", stop_app_handler(app))


def load_resources():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("words")


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logging.info("load_resources start")
#     load_resources()
#     logging.info("load_resources finished")
#     yield


@app.on_event("startup")
async def startup_event():
    logging.info("load_resources start")
    load_resources()
    logging.info("load_resources finished")


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
