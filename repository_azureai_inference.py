import os
from typing import List

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.aio import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI configurations
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")
OPENAI_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")

# openai_client = AsyncAzureOpenAI(azure_endpoint=OPENAI_ENDPOINT,
#                                  api_key=OPENAI_API_KEY,
#                                  api_version=OPENAI_API_VERSION)\\

embedding_client = EmbeddingsClient(
    endpoint=OPENAI_ENDPOINT + '/openai/deployments/' + OPENAI_EMBEDDING_MODEL,
    credential=AzureKeyCredential(OPENAI_API_KEY),
    # Azure OpenAI api-version. See https://aka.ms/azsdk/azure-ai-inference/azure-openai-api-versions
    api_version=OPENAI_API_VERSION
)

chat_client = ChatCompletionsClient(
    endpoint=OPENAI_ENDPOINT + '/openai/deployments/' + OPENAI_CHAT_MODEL,
    credential=AzureKeyCredential(OPENAI_API_KEY),
    api_version=OPENAI_API_VERSION,
    temperature=0.7,
    max_tokens=500,
)


async def get_embedding(text):
    response = await embedding_client.embed(input=[text])

    embedding = response.data[0].embedding

    return embedding


async def get_chat_answer(messages: List[dict]) -> str:
    response = chat_client.complete(
        messages=messages,
    )

    print(f"Total tokens: {response.usage.total_tokens}")
    print(f"Response message content: {response.choices[0].message.content}")

    answer = response.choices[0].message.content
    return answer
