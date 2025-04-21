import os
os.environ['TORCH_DISABLE_CUSTOM_CLASS_PATHS'] = '1'

import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Any
from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TEXT_CHUNK_SIZE,
    TEXT_CHUNK_OVERLAP,
    CHROMA_SETTINGS
)

# Lazy load PyTorch-dependent imports
sentence_transformer = None
def get_sentence_transformer():
    global sentence_transformer
    if sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_transformer

# Load environment variables
load_dotenv()

def process_pdf_to_embeddings(pdf_path: str, collection_name: str = COLLECTION_NAME) -> Chroma:
    """Process PDF and create embeddings with enhanced structure awareness"""
    
    # Initialize the PDF loader
    loader = PyPDFLoader(pdf_path)
    
    # Load PDF pages
    pages = loader.load()
    
    # Create a text splitter that respects document structure
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ".",     # Sentences
            " ",     # Words
            ""       # Characters
        ]
    )
    
    # Process and structure the documents
    processed_chunks = []
    current_section = ""
    current_subsection = ""
    
    for page in pages:
        # Extract content and clean it
        content = page.page_content
        chunks = text_splitter.split_text(content)
        
        for chunk in chunks:
            # Identify section and subsection headers
            lines = chunk.split('\n')
            for line in lines:
                if _is_section_header(line):
                    current_section = line.strip()
                    current_subsection = ""
                elif _is_subsection_header(line):
                    current_subsection = line.strip()
            
            # Create structured chunk with metadata
            processed_chunk = {
                "content": chunk,
                "metadata": {
                    "page": page.metadata.get("page", 0),
                    "section": str(current_section),
                    "subsection": str(current_subsection),
                    "source": str(pdf_path),
                    "chunk_type": str(_determine_chunk_type(chunk)),
                    "requirement_indicators": ",".join(_extract_requirement_indicators(chunk)) or "none"
                }
            }
            processed_chunks.append(processed_chunk)
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Try to clean up any existing embeddings
    try:
        if CHROMA_DB_DIR.exists():
            try:
                import time
                max_retries = 3
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        shutil.rmtree(str(CHROMA_DB_DIR))
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            print(f"Database is locked, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print("Warning: Could not clean up old database. Will attempt to continue...")
            except Exception as e:
                print(f"Warning: Error cleaning up database: {str(e)}. Will attempt to continue...")
    except Exception as e:
        print(f"Warning: Error checking database existence: {str(e)}. Will attempt to continue...")
    
    # Create the vector store
    texts = [chunk["content"] for chunk in processed_chunks]
    metadatas = [chunk["metadata"] for chunk in processed_chunks]
    
    try:
        # Filter any remaining complex metadata types
        for metadata in metadatas:
            for key in list(metadata.keys()):
                if not isinstance(metadata[key], (str, int, float, bool)):
                    metadata[key] = str(metadata[key])
                    
        # Initialize ChromaDB with persistence
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        
        # Add texts to collection with unique IDs
        ids = [f"doc_{i}" for i in range(len(texts))]
        db.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully processed {len(texts)} text chunks from the PDF")
        
        return db
        
    except Exception as e:
        print(f"Error creating vector database: {str(e)}")
        raise

def _is_section_header(text: str) -> bool:
    """Check if text is a main section header"""
    text = text.strip()
    # Match patterns like "4.", "5.", etc. at start of line
    return (
        len(text) > 2 and
        text[0].isdigit() and
        text[1] == '.' and
        not text[2].isdigit()
    )

def _is_subsection_header(text: str) -> bool:
    """Check if text is a subsection header"""
    text = text.strip()
    # Match patterns like "4.1", "5.2", etc. at start of line
    return (
        len(text) > 3 and
        text[0].isdigit() and
        text[1] == '.' and
        text[2].isdigit()
    )

def _determine_chunk_type(text: str) -> str:
    """Determine the type of content in the chunk"""
    text_lower = text.lower()
    
    if any(term in text_lower for term in ["shall", "must", "required", "mandatory"]):
        return "requirement"
    elif any(term in text_lower for term in ["should", "recommended", "may"]):
        return "recommendation"
    elif "note" in text_lower:
        return "note"
    elif any(term in text_lower for term in ["example", "e.g.", "such as"]):
        return "example"
    else:
        return "information"

def _extract_requirement_indicators(text: str) -> List[str]:
    """Extract requirement indicators from text"""
    indicators = []
    text_lower = text.lower()
    
    # Common requirement phrases
    requirement_phrases = [
        "shall",
        "must",
        "required",
        "mandatory",
        "should",
        "ensure that",
        "maintain",
        "implement",
        "establish",
        "define",
        "document"
    ]
    
    # Check for each phrase
    for phrase in requirement_phrases:
        if phrase in text_lower:
            indicators.append(phrase)
    
    return indicators

if __name__ == "__main__":
    # Process the ISO 27001:2022 PDF
    pdf_path = "ISO_27001_2022_ISMS.pdf"
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    print(f"Processing {pdf_path}...")
    db = process_pdf_to_embeddings(pdf_path)
    print("PDF has been processed and stored in ChromaDB")
    print(f"Embeddings are stored in the '{COLLECTION_NAME}' collection")