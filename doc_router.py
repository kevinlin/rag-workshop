from fastapi import APIRouter, UploadFile, File
from ingest import process_document


def doc_router() -> APIRouter:
    router = APIRouter()

    @router.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        # Process the uploaded file
        result = process_document(file)
        return result

    return router