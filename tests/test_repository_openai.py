import pytest

from ingest import read_pdf_content, chunk_text
from repository_openai import get_embedding


@pytest.mark.asyncio
async def test_get_embedding():
    content = await read_pdf_content("../input_docs/open-banking.pdf")
    chunks = chunk_text(content)

    embedding = await get_embedding(chunks[0])
    assert embedding is not None
    print("Embedding created:", embedding)
