from typing import List

from fastapi import APIRouter
from pydantic import BaseModel

from ingest import print_chunks
from repository_azureai_inference import get_chat_answer, get_embedding
from repository_vector_db import search_chunks

# In-memory chat history (global variable for now)
chat_history = []


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


def chat_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=ChatResponse)
    async def get_chat_response(chat_request: ChatRequest):
        question = chat_request.question

        # Append the question to the chat history
        chat_history.append({"role": "user", "content": question})

        # Get embedding of the question
        question_embedding = await get_embedding(question)

        # Retrieve the best 3 chunks from Azure AI Search
        top_chunks = await search_chunks(question, question_embedding, top_k=3)
        print("get_chat_response()->top_chunks:")
        print_chunks(top_chunks)

        # Construct the messages with chat history and context
        messages = construct_messages(chat_history, top_chunks)
        # print(f"get_chat_response()->messages: {messages}")

        # Get the answer from Azure OpenAI
        answer = await get_chat_answer(messages)
        print(f"get_chat_response()->answer: {answer}")

        # Append the assistant's answer to the chat history
        chat_history.append({"role": "assistant", "content": answer})

        return ChatResponse(answer=answer)

    return router


def construct_messages(chat_history: List[dict], chunks: List[str]) -> List[dict]:
    # Combine chat history and context into messages
    messages = chat_history.copy()

    # Add context as system message
    context = "\n\n".join(chunks)
    system_message = {
        "role": "system",
        "content": f"Use the following context to answer the user's question:\n{context}"
    }
    messages.insert(0, system_message)

    messages.insert(0, {"role": "system", "content": "You are an AI assistant."})

    return messages
