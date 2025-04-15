import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb

# Load environment variables
load_dotenv()

def process_pdf_to_embeddings(pdf_path: str, collection_name: str):
    # Load and split the PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(pages)
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create and persist a Chroma vector store
    db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=collection_name
    )
    
    # Persist the database
    db.persist()
    return db

if __name__ == "__main__":
    # Process the ISO 27001:2022 PDF
    pdf_path = "ISO_27001_2022_ISMS.pdf"
    collection_name = "iso27001_2022"
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    print(f"Processing {pdf_path}...")
    db = process_pdf_to_embeddings(pdf_path, collection_name)
    print("PDF has been processed and stored in ChromaDB")
    print(f"Embeddings are stored in the '{collection_name}' collection")