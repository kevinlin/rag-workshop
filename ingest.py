import os
import uuid
from typing import List
import openai

import httpx
import nltk
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (SimpleField, SearchableField, SearchIndex, VectorSearch,
                                                   VectorSearchAlgorithmConfiguration)
from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile
from nltk import sent_tokenize
from pydantic import BaseModel
from starlette.exceptions import HTTPException

load_dotenv()

chunk_size = int(os.getenv("CHUNK_SIZE", "500"))

# Azure OpenAI configurations
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_ENGINE = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")

openai.api_type = "azure"
openai.api_base = OPENAI_ENDPOINT
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY

# Azure AI Search configurations
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

# Initialize NLTK
nltk.download('punkt', quiet=True)


class Document(BaseModel):
    id: str
    content: str
    embedding: List[float]


def process_document(file: UploadFile):
    # Read the file content
    content = file.file.read()
    filename = file.filename.lower()

    # Extract text based on file type
    if filename.endswith('.pdf'):
        text = extract_text_from_pdf(content)
    elif filename.endswith('.txt'):
        text = content.decode('utf-8')
    else:
        return {"error": "Unsupported file type. Please upload a PDF or TXT file."}

    # Chunk the text
    chunks = chunk_text(text, chunk_size)

    # Generate embeddings
    embeddings = get_embeddings(chunks)

    # Create or update the search index
    create_search_index()

    # Upload to Azure Cognitive Search
    upload_chunks_to_search(chunks, embeddings)

    return {"message": "File processed and uploaded successfully."}


def extract_text_from_pdf(pdf_bytes):
    from io import BytesIO
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text


def chunk_text(text: str, max_chunk_size=500) -> list:
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embeddings.append(embedding)

    return embeddings


def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model=EMBEDDING_ENGINE
    )
    embedding = response.data[0].embedding
    return embedding


async def convert_to_embedding(text: str):
    headers = {
        "Content-Type": "application/json",
        "api-key": OPENAI_API_KEY
    }
    data = {
        "input": text
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(OPENAI_EMBEDDING_ENDPOINT, headers=headers, json=data)
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]

    return embedding


def create_search_index():
    # Check if the index exists
    try:
        index_client.get_index(name=SEARCH_INDEX_NAME)
    except Exception:
        # Define the index schema
        fields = [
            SimpleField(name="id", type="Edm.String", key=True),
            SearchableField(name="content", type="Edm.String"),
            SimpleField(
                name="embedding",
                type="Collection(Edm.Single)",
                searchable=True,
                dimensions=1536,
                vector_search_configuration="default"
            )
        ]
        vector_search = VectorSearch(
            algorithm_configurations=[VectorSearchAlgorithmConfiguration(name="default", kind="hnsw")]
        )

        index = SearchIndex(
            name=SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search
        )
        # Create the index
        index_client.create_index(index)


def upload_chunks_to_search(chunks, embeddings):
    documents = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            'id': str(uuid.uuid4()),
            'content': chunk,
            'embedding': embedding
        }
        documents.append(doc)

    # Upload documents to Azure Cognitive Search
    results = search_client.upload_documents(documents)
    print(f"Uploaded documents: {results[0]}")


# Old code
async def read_pdf_content(file_path: str) -> str:
    with open(file_path, "rb") as file:
        content = await file.read()
        filename = file.filename.lower()

        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(content)
        elif filename.endswith('.txt'):
            text = content.decode('utf-8')
        else:
            print(f"Unsupported file type: {filename}. Please upload a PDF or TXT file.")
            return {"error": "Unsupported file type. Please upload a PDF or TXT file."}

        print(f"Extracted text from {filename}")
        return text


async def retrieve_best_chunks(question: str):
    # Query Azure AI Search with the question
    results = search_client.search(search_text=question, top=3)

    # Extract the content of the top 3 chunks
    chunks = [doc["content"] for doc in results]

    return chunks
