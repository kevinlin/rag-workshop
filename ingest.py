import os
import uuid
from typing import List

import nltk
from dotenv import load_dotenv
from fastapi import UploadFile
from nltk import word_tokenize, sent_tokenize
from pydantic import BaseModel
import tiktoken

from repository_openai import get_embedding
from repository_vector_db import ai_search_client, create_search_index

load_dotenv()

model_name = os.getenv("AZURE_OPENAI_CHAT_MODEL", "t5-small")
chunk_size = int(os.getenv("CHUNK_SIZE", "500"))

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
    chunks = chunk_text(text, chunk_size, model_name)

    # Generate embeddings
    embeddings = await get_embeddings(chunks)

    # Create or update the search index
    await create_search_index()

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


def chunk_text(text: str, max_tokens=500, model_name='gpt-4o-mini') -> list:
    # Initialize the tokenizer
    encoding = tiktoken.encoding_for_model(model_name)

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''

    for sentence in sentences:
        # Estimate tokens in the current chunk and the new sentence
        current_tokens = len(encoding.encode(current_chunk))
        sentence_tokens = len(encoding.encode(sentence))

        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += ' ' + sentence if current_chunk else sentence
        else:
            if current_chunk:
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


# Used for testing only
def read_pdf_content(file_path: str) -> str:
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

        return text
