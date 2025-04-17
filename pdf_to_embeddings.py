import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

def process_pdf_to_embeddings(pdf_path: str, collection_name: str) -> Chroma:
    """Process PDF and create embeddings with enhanced structure awareness"""
    
    # Initialize the PDF loader
    loader = PyPDFLoader(pdf_path)
    
    # Load PDF pages
    pages = loader.load()
    
    # Create a text splitter that respects document structure
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
                    "section": current_section,
                    "subsection": current_subsection,
                    "source": pdf_path,
                    "chunk_type": _determine_chunk_type(chunk),
                    "requirement_indicators": _extract_requirement_indicators(chunk)
                }
            }
            processed_chunks.append(processed_chunk)
    
    # Clean up any existing embeddings
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create and persist the vector store
    texts = [chunk["content"] for chunk in processed_chunks]
    metadatas = [chunk["metadata"] for chunk in processed_chunks]
    
    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    # Persist the database
    db.persist()
    
    return db

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
    collection_name = "iso27001_2022"
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    print(f"Processing {pdf_path}...")
    db = process_pdf_to_embeddings(pdf_path, collection_name)
    print("PDF has been processed and stored in ChromaDB")
    print(f"Embeddings are stored in the '{collection_name}' collection")