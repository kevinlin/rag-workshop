import pytest

from ingest import read_pdf_content, convert_to_embedding, chunk_text, save_to_vector_db


def test_read_pdf_content():
    read_pdf_content("../input_docs/open-banking.pdf")


def test_chunk_text():
    file_content = read_pdf_content("../input_docs/open-banking.pdf")
    chunks = chunk_text(file_content)
    _print_chunks(chunks)


@pytest.mark.asyncio
async def test_convert_to_embedding():
    chunks = chunk_text(read_pdf_content("../input_docs/open-banking.pdf"))

    embedding = await convert_to_embedding(chunks[0])
    assert embedding is not None
    print("Embedding created:", embedding)


@pytest.mark.asyncio
async def test_save_to_vector_db():
    chunks = chunk_text(read_pdf_content("../input_docs/open-banking.pdf"))
    for chunk in chunks:
        await save_to_vector_db(chunk)


def _print_chunks(chunks: list):
    for index, chunk in enumerate(chunks):
        preview = ' '.join(chunk.split()[:100]) + '...'  # Get the first few words
        print(f"Chunk {index + 1}:")
        print(f"Length: {len(chunk)}")
        print(f"Content: {preview}")
        print("-" * 40)
