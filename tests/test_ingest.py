from io import BytesIO

import tiktoken

import pytest
from starlette.datastructures import UploadFile

from ingest import read_pdf_content, chunk_text, process_document, print_chunks


def test_read_pdf_content():
    file_content = read_pdf_content("../input_docs/open-banking.pdf")
    print(f"Extracted text: {file_content}")


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

