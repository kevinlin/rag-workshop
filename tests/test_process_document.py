# test_ingest.py

from io import BytesIO

from starlette.datastructures import UploadFile

from ingest import process_document


def test_process_document_txt():
    # Path to your test TXT file
    txt_file_path = 'test_document.txt'

    # Read the content of the test file
    with open(txt_file_path, 'rb') as f:
        file_content = f.read()

    # Create a mock UploadFile
    upload_file = UploadFile(filename='test_document.txt', file=BytesIO(file_content))

    # Call the process_document function
    result = process_document(upload_file)

    # Print the result
    print(result)
    assert result['message'] == 'File processed and uploaded successfully.'


def test_process_document_pdf():
    # Path to your test PDF file
    pdf_file_path = 'test_document.pdf'

    # Read the content of the test file
    with open(pdf_file_path, 'rb') as f:
        file_content = f.read()

    # Create a mock UploadFile
    upload_file = UploadFile(filename='test_document.pdf', file=BytesIO(file_content))

    # Call the process_document function
    result = process_document(upload_file)

    # Print the result
    print(result)
    assert result['message'] == 'File processed and uploaded successfully.'


if __name__ == '__main__':
    test_process_document_txt()
    test_process_document_pdf()
