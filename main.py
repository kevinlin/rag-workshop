from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from chat_router import chat_router


def init():
    app = FastAPI()

    app.include_router(chat_router())

    return app


app = init()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
