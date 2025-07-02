import os
import speech_recognition as sr
import pyttsx3
import winsound  # for beep
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# ---------------- ENV SETUP ----------------
load_dotenv()
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PERSIST_PATH = "faiss_crystal_index"

# ---------------- VOICE SETUP ----------------
recognizer = sr.Recognizer()
tts = pyttsx3.init()

# ---------------- EMBEDDINGS ----------------
embedding_model = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)

# ---------------- FAISS LOAD ----------------
if not os.path.exists(f"{PERSIST_PATH}/index.faiss"):
    raise FileNotFoundError("❌ FAISS index not found. Run your ingest script first.")

vectordb = FAISS.load_local(
    PERSIST_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# ---------------- OPENROUTER LLM ----------------
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",  
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",

)



# ---------------- CONVERSATIONAL CHAIN ----------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=False
)

chat_history = []

# ---------------- VOICE FUNCTIONS ----------------
def beep():
    winsound.Beep(1000, 300)  # frequency, duration in ms

def listen():
    with sr.Microphone() as source:
        print("🎙️ Speak now (after the beep)...")
        beep()
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        print(f"🗣️ You: {query}")
        return query
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."

def speak(text):
    print(f"🤖 Crystal Bot: {text}")
    tts.say(text)
    tts.runAndWait()

# ---------------- MAIN LOOP ----------------
def main():
    # Intro message
    intro_message = (
        "Hello! I'm the Crystal AI Assistant. "
        "I can answer questions about our logistics services, certifications, warehouses, and more. "
        "Just ask your question after the beep. Say 'exit' anytime to quit."
    )
    speak(intro_message)

    print("🚀 Crystal AI Assistant Ready!")
    while True:
        query = listen()
        if query.lower() in ["exit", "quit", "stop"]:
            speak("Goodbye! Have a great day.")
            break

        response = qa_chain.invoke({"question": query, "chat_history": chat_history})
        answer = response["answer"]
        speak(answer)

        # Add to memory
        chat_history.append((query, answer))

if __name__ == "__main__":
    main()
