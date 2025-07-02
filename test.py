import os
from typing import List
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

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
    description="Simple Q&A assistant powered by RAG",
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
class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    response: str

# ---------------- GLOBAL VARIABLES ----------------
qa_chain = None
embedding_model = None
vectordb = None
chat_history = []

# ---------------- INITIALIZATION ----------------
@app.on_event("startup")
async def startup_event():
    global qa_chain, embedding_model, vectordb
    
    try:
        logger.info("Initializing Crystal AI Assistant...")
        
        # Initialize embeddings
        embedding_model = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
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
        
        # Initialize LLM
        llm = ChatOpenAI(
            model_name="meta-llama/llama-4-maverick:free",
            openai_api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        logger.info("✅ LLM initialized")
        
        # Initialize conversational chain (same as app.py)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=False
        )
        logger.info("✅ Conversational chain ready")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise e

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
        "message": "Crystal AI Assistant is running",
        "qa_chain_ready": qa_chain is not None,
        "vectordb_size": vectordb.index.ntotal if vectordb else 0
    }

@app.post("/chat", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    """Simple Q&A endpoint - same logic as app.py"""
    global chat_history
    
    try:
        if qa_chain is None:
            raise HTTPException(status_code=503, detail="AI Assistant not ready")
        
        # Get response from the chain (same as app.py)
        response = qa_chain.invoke({"question": query.message, "chat_history": chat_history})
        answer = response["answer"]
        
        # Add to memory (same as app.py)
        chat_history.append((query.message, answer))
        
        return QueryResponse(response=answer)
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global chat_history
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process the query (same logic as app.py)
            query = message_data["message"]
            
            try:
                response = qa_chain.invoke({"question": query, "chat_history": chat_history})
                answer = response["answer"]
                
                # Add to memory (same as app.py)
                chat_history.append((query, answer))
                
                # Send response back
                await manager.send_personal_message(
                    json.dumps({
                        "response": answer,
                        "type": "answer"
                    }),
                    websocket
                )
            except Exception as e:
                await manager.send_personal_message(
                    json.dumps({
                        "response": "Sorry, I couldn't process your question. Please try again.",
                        "type": "error"
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