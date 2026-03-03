import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import uuid

class DocumentService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

    async def process_pdf(self, file_bytes: bytes, filename: str):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        documents = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue
                
            # Create chunks for this page
            chunks = self.text_splitter.split_text(text)
            
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "doc_id": str(uuid.uuid4())
                    }
                ))
        
        return documents
