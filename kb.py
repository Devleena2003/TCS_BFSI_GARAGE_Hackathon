import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader # Loads all PDFs from a directory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

load_dotenv() 


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
genai.configure(api_key=GOOGLE_API_KEY)

PDF_DIRECTORY = "data" 
VECTOR_STORE_PATH = "faiss_index"

def create_vector_store():
    print(f"Loading PDFs from {PDF_DIRECTORY}...")
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    documents = loader.load()
    if not documents:
        print("No documents found in the directory.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150 
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} text chunks.")


    if not texts:
         print("No text chunks generated after splitting.")
         return

    print("Generating embeddings using Google Generative AI...")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 

    print("Creating FAISS vector store...")
    try:
        vector_store = FAISS.from_documents(texts, embeddings)
        print("Vector store created successfully.")
        print(f"Saving vector store to {VECTOR_STORE_PATH}...")
        vector_store.save_local(VECTOR_STORE_PATH)
        print("Vector store saved.")
    except Exception as e:
        print(f"Error creating or saving vector store: {e}")
       

if __name__ == "__main__":
    if not os.path.exists(PDF_DIRECTORY) or not os.listdir(PDF_DIRECTORY):
         print(f"Error: PDF directory '{PDF_DIRECTORY}' does not exist or is empty.")
         print("Please create the 'data' directory and add your policy PDFs.")
    else:
        create_vector_store()