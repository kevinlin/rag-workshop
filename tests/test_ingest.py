from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from ingest import read_pdf_content, chunk_text, process_document


@pytest.mark.asyncio
def test_read_pdf_content():
    read_pdf_content("../input_docs/open-banking.pdf")


def test_chunk_text():
    file_content = read_pdf_content("../input_docs/open-banking.pdf")
    chunks = chunk_text(file_content)
    print_chunks(chunks)


@pytest.mark.asyncio
async def test_process_document_txt():
    # Path to your test TXT file
    txt_file_path = 'test_document.txt'

    # Read the content of the test file
    with open(txt_file_path, 'rb') as f:
        file_content = f.read()

    # Create a mock UploadFile
    upload_file = UploadFile(filename='test_document.txt', file=BytesIO(file_content))

    # Call the process_document function
    result = await process_document(upload_file)

    # Print the result
    print(result)
    assert result['message'] == 'File processed and uploaded successfully.'


@pytest.mark.asyncio
async def test_process_document_pdf():
    # Path to your test PDF file
    pdf_file_path = 'test_document.pdf'

    # Read the content of the test file
    with open(pdf_file_path, 'rb') as f:
        file_content = f.read()

    # Create a mock UploadFile
    upload_file = UploadFile(filename='test_document.pdf', file=BytesIO(file_content))

    # Call the process_document function
    result = await process_document(upload_file)

    # Print the result
    print(result)
    assert result['message'] == 'File processed and uploaded successfully.'


def print_chunks(chunks: list):
    for index, chunk in enumerate(chunks):
        preview = ' '.join(chunk.split()[:100]) + '...'  # Get the first few words
        print(f"Chunk {index + 1}:")
        print(f"Length: {len(chunk)}")
        print(f"Content: {preview}")
        print("-" * 40)
