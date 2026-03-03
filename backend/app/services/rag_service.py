from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from backend.app.core.config import settings

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY
        )

    def get_vectorstore(self, session_id: str):
        return Chroma(
            collection_name=f"session_{session_id}",
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DB_DIR
        )

    async def add_documents(self, session_id: str, documents: list):
        vectorstore = self.get_vectorstore(session_id)
        await vectorstore.aadd_documents(documents)

    async def query(self, session_id: str, query: str, history: list):
        vectorstore = self.get_vectorstore(session_id)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        template = """You are a helpful assistant. Answer only using the provided context. 
        If the answer is not in the context, say 'The answer was not found in the uploaded documents.' 
        Do not hallucinate.
        
        Context: {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Retrieve relevant documents
        docs = await retriever.ainvoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Generate answer
        chain = prompt | self.llm
        response = await chain.ainvoke({"context": context, "question": query})
        
        # Extract unique sources
        sources = []
        seen = set()
        for d in docs:
            source_key = (d.metadata.get("source"), d.metadata.get("page"))
            if source_key not in seen:
                sources.append({
                    "document": d.metadata.get("source"),
                    "page": d.metadata.get("page")
                })
                seen.add(source_key)
        
        return response.content, sources
