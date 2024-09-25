from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from requests import request
import os

from openai.lib.azure import AsyncAzureOpenAI

from llm import get_openai_answer

AZURE_OPENAI_ENDPOINT = "https://<your-endpoint>.openai.azure.com/openai/deployments/<your-deployment>/completions?api-version=2022-12-01"
AZURE_OPENAI_API_KEY = "<your-api-key>"


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


def chat_router() -> APIRouter:
    router = APIRouter()

    @router.post("/v1/api/chats", response_model=ChatResponse)
    async def get_chat_response(question: ChatRequest):
        answer = await get_openai_answer(question.question)

        return ChatResponse(answer=answer)

    return router
