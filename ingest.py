from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os

def create_faiss_index(txt_path: str, save_path: str = "faiss_crystal_index"):
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(documents, embeddings)

    faiss_index.save_local(save_path)
    print(f"✅ FAISS index saved at '{save_path}'")

if __name__ == "__main__":
    create_faiss_index("data/extracted_text.txt")
