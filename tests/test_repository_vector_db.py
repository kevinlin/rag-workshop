import pytest

from repository_openai import get_embedding
from repository_vector_db import create_search_index, search_chunks
from tests.test_ingest import print_chunks


@pytest.mark.asyncio
async def test_create_search_index():
    await create_search_index()


@pytest.mark.asyncio
async def test_search_chunks():
    question = "What is open banking?"
    question_embedding = await get_embedding(question)
    chunks = await search_chunks(question, question_embedding)
    print_chunks(chunks)
