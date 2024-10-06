import os
from openai.lib.azure import AsyncAzureOpenAI
from typing import List

from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI configurations
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_API_DEPLOYMENT")

openai_client = AsyncAzureOpenAI(azure_endpoint=OPENAI_ENDPOINT,
                                 api_key=OPENAI_API_KEY,
                                 api_version=OPENAI_API_VERSION)


async def get_embedding(text):
    response = await openai_client.embeddings.create(
        input=text,
        model=OPENAI_EMBEDDING_MODEL
    )
    embedding = response.data[0].embedding

    return embedding


async def get_chat_answer(messages: List[dict]) -> str:
    response = await openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
        top_p=0.95,
        n=1,
    )

    print(f"Total tokens: {response.usage.total_tokens}")
    print(f"Response message role: {response.choices[0].message.role}")

    answer = response.choices[0].message.content
    return answer
