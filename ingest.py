import os
import re

import httpx
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


def ingestion():
    # Define the relative path to the PDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "input_docs", "open-banking.pdf")

    try:
        # Attempt to open and read the PDF file
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)
            text = ""

            # Extract text from each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()

            # Output the extracted text
            print(text)

    # Exception handling for various potential issues

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except PermissionError:
        print(f"Error: You do not have permission to access the file '{pdf_path}'.")
    except Exception as e:
        # General exception handler to catch other errors (e.g., PDF parsing issues)
        print(f"An error occurred while reading the PDF: {e}")
