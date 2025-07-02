import os
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- ENV SETUP ----------------
load_dotenv()
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PERSIST_PATH = "faiss_crystal_index"

# ---------------- FASTAPI SETUP ----------------
app = FastAPI(
    title="Crystal AI Assistant API",
    description="Intelligent logistics assistant powered by RAG and LLM with voice interaction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    sources: List[str] = []
    session_id: str
    confidence_score: Optional[float] = None

# ---------------- GLOBAL VARIABLES ----------------
qa_chain = None
embedding_model = None
vectordb = None
chat_sessions: Dict[str, List[tuple]] = {}

# ---------------- IMPROVED PROMPT TEMPLATE ----------------
CUSTOM_PROMPT = PromptTemplate(
    template="""You are Crystal AI Assistant, an intelligent logistics assistant. Use the following context to answer the user's question accurately and comprehensively.

Context from knowledge base:
{context}

Chat History:
{chat_history}

Current Question: {question}

Instructions:
1. Use ONLY the information provided in the context above
2. If the context doesn't contain enough information to answer fully, say so explicitly
3. Be specific and detailed in your response
4. If you find conflicting information, mention it
5. Always ground your answer in the provided context
6. For logistics questions, provide actionable insights when possible

Answer:""",
    input_variables=["context", "chat_history", "question"]
)

# ---------------- INITIALIZATION ----------------
@app.on_event("startup")
async def startup_event():
    global qa_chain, embedding_model, vectordb
    
    try:
        logger.info("Initializing Crystal AI Assistant...")
        
        # Initialize embeddings with better parameters
        embedding_model = HuggingFaceEmbeddings(
            model_name=HF_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Important for similarity search
        )
        logger.info("✅ Embeddings model loaded")
        
        # Load FAISS index
        if not os.path.exists(f"{PERSIST_PATH}/index.faiss"):
            logger.error("❌ FAISS index not found. Run your ingest script first.")
            raise FileNotFoundError("FAISS index not found")
        
        vectordb = FAISS.load_local(
            PERSIST_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        logger.info(f"✅ FAISS vectorstore loaded with {vectordb.index.ntotal} documents")
        
        # Initialize LLM with better parameters
        llm = ChatOpenAI(
            model_name="mistralai/mistral-7b-instruct",
            openai_api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,  # Lower temperature for more consistent responses
            max_tokens=500,
            request_timeout=60
        )
        logger.info("✅ LLM initialized")
        
        # Initialize conversational chain with custom prompt and better retriever
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 5,  # Retrieve top 5 most relevant documents
                    "fetch_k": 20,  # Fetch more candidates for better selection
                }
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
            verbose=True  # Enable verbose logging for debugging
        )
        logger.info("✅ Conversational chain ready")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise e

# ---------------- HELPER FUNCTIONS ----------------
def calculate_relevance_score(query: str, documents: List[Document]) -> float:
    """Calculate a simple relevance score based on keyword overlap"""
    if not documents:
        return 0.0
    
    query_words = set(query.lower().split())
    total_score = 0.0
    
    for doc in documents:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words.intersection(doc_words))
        total_score += overlap / len(query_words) if query_words else 0
    
    return total_score / len(documents)

def debug_retrieval(query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    """Debug function to analyze retrieval quality"""
    debug_info = {
        "query": query,
        "num_retrieved": len(retrieved_docs),
        "docs_preview": []
    }
    
    for i, doc in enumerate(retrieved_docs[:3]):  # Show first 3 docs
        debug_info["docs_preview"].append({
            "doc_index": i,
            "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
            "content_length": len(doc.page_content)
        })
    
    return debug_info

# ---------------- API ENDPOINTS ----------------
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Crystal AI Assistant with voice interaction is running",
        "embeddings_loaded": embedding_model is not None,
        "vectordb_loaded": vectordb is not None,
        "vectordb_size": vectordb.index.ntotal if vectordb else 0,
        "qa_chain_ready": qa_chain is not None,
        "voice_features": "enabled"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint with improved retrieval"""
    try:
        if qa_chain is None:
            raise HTTPException(status_code=503, detail="AI Assistant not ready")
        
        # Get or create chat history for session
        if chat_message.session_id not in chat_sessions:
            chat_sessions[chat_message.session_id] = []
        
        chat_history = chat_sessions[chat_message.session_id]
        
        # First, test retrieval directly to debug
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        # Get relevant documents
        relevant_docs = retriever.get_relevant_documents(chat_message.message)
        
        # Debug retrieval
        debug_info = debug_retrieval(chat_message.message, relevant_docs)
        logger.info(f"Retrieval debug: {debug_info}")
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(chat_message.message, relevant_docs)
        
        # Get response from the chain
        response = qa_chain.invoke({
            "question": chat_message.message,
            "chat_history": chat_history
        })
        
        answer = response["answer"]
        sources = []
        
        # Extract sources if available
        if "source_documents" in response and response["source_documents"]:
            sources = []
            for doc in response["source_documents"]:
                source_info = doc.metadata.get("source", "Unknown")
                if "page" in doc.metadata:
                    source_info += f" (Page {doc.metadata['page']})"
                sources.append(source_info)
        
        # Update chat history
        chat_history.append((chat_message.message, answer))
        
        # Keep only last 10 exchanges to manage memory
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
            chat_sessions[chat_message.session_id] = chat_history
        
        return ChatResponse(
            response=answer,
            sources=list(set(sources)),  # Remove duplicates
            session_id=chat_message.session_id,
            confidence_score=relevance_score
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/debug/search")
async def debug_search(query: str, k: int = 5):
    """Debug endpoint to test vector search directly"""
    if vectordb is None:
        raise HTTPException(status_code=503, detail="Vector database not ready")
    
    try:
        # Perform similarity search
        docs = vectordb.similarity_search(query, k=k)
        
        # Also get similarity scores
        docs_with_scores = vectordb.similarity_search_with_score(query, k=k)
        
        result = {
            "query": query,
            "num_results": len(docs),
            "results": []
        }
        
        for i, (doc, score) in enumerate(docs_with_scores):
            result["results"].append({
                "rank": i + 1,
                "similarity_score": float(score),
                "content_preview": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                "metadata": doc.metadata,
                "content_length": len(doc.page_content)
            })
        
        return result
        
    except Exception as e:
        logger.error(f"Debug search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Debug search failed: {str(e)}")

@app.get("/vectordb/info")
async def get_vectordb_info():
    """Get information about the vector database"""
    if vectordb is None:
        raise HTTPException(status_code=503, detail="Vector database not ready")
    
    return {
        "total_documents": vectordb.index.ntotal,
        "embedding_dimension": vectordb.index.d,
        "index_type": type(vectordb.index).__name__,
        "embedding_model": HF_MODEL_NAME
    }

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    if session_id not in chat_sessions:
        return {"history": []}
    
    history = []
    for user_msg, bot_msg in chat_sessions[session_id]:
        history.extend([
            {"message": user_msg, "type": "user"},
            {"message": bot_msg, "type": "bot"}
        ])
    
    return {"history": history}

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    return {"message": f"Session {session_id} cleared"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_sessions": len(chat_sessions),
        "active_sessions": len([s for s in chat_sessions.values() if len(s) > 0]),
        "total_exchanges": sum(len(history) for history in chat_sessions.values()),
        "vectordb_docs": vectordb.index.ntotal if vectordb else 0,
        "voice_enabled": True
    }

# ---------------- WEBSOCKET SUPPORT ----------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the message
            chat_message = ChatMessage(
                message=message_data["message"],
                session_id=session_id
            )
            
            # Get response (using the improved chat logic)
            response = await chat_endpoint(chat_message)
            
            # Send response back
            await manager.send_personal_message(
                json.dumps({
                    "response": response.response,
                    "sources": response.sources,
                    "session_id": session_id,
                    "confidence_score": response.confidence_score
                }),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )