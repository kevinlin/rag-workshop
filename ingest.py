import os
import re
import uuid

import httpx
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from fastapi import HTTPException
from pypdf import PdfReader
from starlette.exceptions import HTTPException

load_dotenv()


def read_pdf_content(file_path: str) -> str:
    try:
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

            return text

    except FileNotFoundError as e:
        print(f"Error: The file '{file_path}' was not found.")
        raise HTTPException(status_code=404, detail=f"The file '{file_path}' was not found: {e}")
    except PermissionError as e:
        print(f"Error: You do not have permission to access the file '{file_path}'.")
        raise HTTPException(status_code=404, detail=f"You do not have permission to access the file '{file_path}': {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {e}")


def chunk_text(text: str) -> list:
    # Split text into sentences
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


async def convert_to_embedding(text: str):
    openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    embedding_model_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")

    headers = {
        "Content-Type": "application/json",
        "api-key": openai_api_key
    }
    data = {
        "input": text
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(embedding_model_endpoint, headers=headers, json=data)
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]

    return embedding

async def save_to_vector_db(text: str):
    # Initialize the SearchClient
    search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    search_api_key = os.getenv("AZURE_SEARCH_API_KEY")

    search_client = SearchClient(
        endpoint=search_service_endpoint,
        index_name=search_index_name,
        credential=AzureKeyCredential(search_api_key)
    )

    # Convert text to embedding
    embedding = await convert_to_embedding(text)

    # Create a document with the embedding
    document = {
        "id": str(uuid.uuid4()),
        "content": text,
        "embedding": embedding
    }

    # Upload the document to the Azure AI Search index
    result = search_client.upload_documents(documents=[document])
    if not result[0].succeeded:
        raise HTTPException(status_code=500, detail="Failed to upload document to Azure AI Search")

    print("Document uploaded successfully")