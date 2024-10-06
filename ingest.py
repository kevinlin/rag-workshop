import os
import uuid
from lib2to3.pgen2.token import NEWLINE
from typing import List

import httpx
import nltk
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery, VectorizedQuery
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (HnswAlgorithmConfiguration,
                                                   SearchField,
                                                   SearchFieldDataType,
                                                   SearchIndex,
                                                   SearchableField,
                                                   SimpleField,
                                                   VectorSearch,
                                                   VectorSearchProfile)
from dotenv import load_dotenv
from fastapi import UploadFile
from nltk import word_tokenize
from openai.lib.azure import AsyncAzureOpenAI
from pydantic import BaseModel
import numpy as np

load_dotenv()

chunk_size = int(os.getenv("CHUNK_SIZE", "500"))

# Azure OpenAI configurations
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_ENGINE = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
OPENAI_EMBEDDING_ENDPOINT = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")

openai_client = AsyncAzureOpenAI(azure_endpoint=OPENAI_ENDPOINT,
                                 api_version=OPENAI_API_VERSION,
                                 api_key=OPENAI_API_KEY)

# Azure AI Search configurations
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

ai_search_client = SearchClient(
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
    filename: str
    content: str
    embedding: List[float]


async def process_document(file: UploadFile):
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
    embeddings = await get_embeddings(chunks)

    # Create or update the search index
    # await create_search_index()

    # Upload to Azure Cognitive Search
    await upload_chunks_to_search(filename, chunks, embeddings)

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
    sentences = word_tokenize(text)
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


async def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        embedding = await get_embedding(chunk)
        embeddings.append(embedding)

    return embeddings


async def get_embedding(text):
    response = await openai_client.embeddings.create(
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


async def create_search_index():
    # Check if the index exists
    try:
        await index_client.get_index(name=SEARCH_INDEX_NAME)
    except Exception:
        # Define the index schema
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                hidden=False,
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="default"  # use default `myHnswProfile`
            ),
        ]
        vector_search = VectorSearch(
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="default")],
            algorithms=[HnswAlgorithmConfiguration(name="default")]
        )

        index = SearchIndex(
            name=SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search
        )

        # Create the index
        search_index = await index_client.create_index(index)
        print(f"Created index: {search_index.name}")


async def upload_chunks_to_search(filename, chunks, embeddings):
    documents = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            'id': str(uuid.uuid4()),
            'filename': filename,
            'content': chunk,
            'embedding': embedding
        }
        documents.append(doc)

    # Upload documents to Azure Cognitive Search
    results = await ai_search_client.upload_documents(documents)
    print(f"Uploaded documents: {results[0]}")


# Add this function to ingest.py
async def search_chunks(question: str, top_k: int = 3) -> List[str]:
    # Get embedding of the question
    question_embedding = await get_embedding(question)

    # Prepare the vector query
    vector = VectorizedQuery(
        vector=question_embedding,
        fields="embedding",
        k_nearest_neighbors=5,
    )

    # Perform vector search in Azure Cognitive Search
    search_results = await ai_search_client.search(
        search_text=question,
        vector_queries=[vector],
        top=top_k,
        select="filename,content",
    )

    # Extract the content from the search results
    chunks = []
    async for result in search_results:
        filename = result.get('filename', 'Unknown Filename')
        chunks.append(filename + "->\n" + result['content'])
    return chunks


# Used for testing only
async def read_pdf_content(file_path: str) -> str:
    with open(file_path, "rb") as file:
        content = file.read()
        filename = file_path.lower()

        # Extract text based on file type
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(content)
        elif filename.endswith('.txt'):
            text = content.decode('utf-8')
        else:
            print(f"Unsupported file type: {filename}. Please upload a PDF or TXT file.")
            return f"Unsupported file type: {filename}. Please upload a PDF or TXT file."

        print(f"Extracted text from {filename}")
        return text
