import uvicorn
from fastapi import FastAPI

from chat_router import chat_router
from doc_router import doc_router


def init():
    app = FastAPI()

    app.include_router(doc_router(), prefix="/v1/api/docs")
    app.include_router(chat_router(), prefix="/v1/api/chat")

    return app


app = init()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
