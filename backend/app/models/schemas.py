from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    query: str
    history: List[Message] = []

class Source(BaseModel):
    document: str
    page: int

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]

class FileStatus(BaseModel):
    id: str
    name: str
    status: str

class UploadResponse(BaseModel):
    session_id: str
    files: List[FileStatus]
