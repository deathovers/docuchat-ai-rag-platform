from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import uuid
from backend.app.models.schemas import ChatRequest, ChatResponse, UploadResponse, FileStatus
from backend.app.services.document_service import DocumentService
from backend.app.services.rag_service import RAGService

router = APIRouter()
doc_service = DocumentService()
rag_service = RAGService()

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...), session_id: str = None):
    if not session_id:
        session_id = str(uuid.uuid4())
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per session")

    processed_files = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
            
        content = await file.read()
        documents = await doc_service.process_pdf(content, file.filename)
        await rag_service.add_documents(session_id, documents)
        
        processed_files.append(FileStatus(
            id=str(uuid.uuid4()),
            name=file.filename,
            status="processed"
        ))
        
    return UploadResponse(session_id=session_id, files=processed_files)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    answer, sources = await rag_service.query(
        request.session_id, 
        request.query, 
        request.history
    )
    return ChatResponse(answer=answer, sources=sources)
