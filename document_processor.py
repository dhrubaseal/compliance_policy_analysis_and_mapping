import os
import hashlib
from typing import Dict, List, Optional, Union
from pathlib import Path
from diskcache import Cache
import json
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, cache_dir: str = "./cache"):
        self.cache = Cache(cache_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """Process a document with caching support"""
        file_path = str(file_path)
        file_hash = self._calculate_file_hash(file_path)
        
        # Try to get from cache first
        cached_result = self.cache.get(file_hash)
        if cached_result is not None:
            return cached_result
            
        # Process the file based on its type
        result = self._process_file(file_path)
        
        # Cache the result
        self.cache.set(file_hash, result)
        return result
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate a hash of the file content for caching"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _process_file(self, file_path: str) -> Dict[str, any]:
        """Process different file types"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return self._process_csv(file_path)
        elif ext == '.pdf':
            return self._process_pdf(file_path)
        elif ext in ['.doc', '.docx']:
            return self._process_word(file_path)
        elif ext == '.txt':
            return self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _process_csv(self, file_path: str) -> Dict[str, any]:
        """Process CSV files"""
        df = pd.read_csv(file_path)
        return {
            "type": "csv",
            "content": df.to_dict('records'),
            "metadata": {
                "columns": list(df.columns),
                "row_count": len(df)
            }
        }
    
    def _process_pdf(self, file_path: str) -> Dict[str, any]:
        """Process PDF files"""
        reader = PdfReader(file_path)
        content = []
        metadata = {}
        
        # Extract text and metadata from each page
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            content.append({
                "page": i + 1,
                "text": text,
                "layout": self._analyze_page_layout(text)
            })
        
        if reader.metadata:
            metadata = {k.lower(): v for k, v in reader.metadata.items()}
        
        return {
            "type": "pdf",
            "content": content,
            "metadata": metadata
        }
    
    def _process_word(self, file_path: str) -> Dict[str, any]:
        """Process Word documents"""
        doc = Document(file_path)
        content = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                content.append({
                    "text": para.text,
                    "style": para.style.name
                })
        
        return {
            "type": "docx",
            "content": content,
            "metadata": {
                "sections": len(doc.sections),
                "paragraphs": len(doc.paragraphs)
            }
        }
    
    def _process_text(self, file_path: str) -> Dict[str, any]:
        """Process text file and extract controls"""
        loader = TextLoader(file_path)
        document = loader.load()
        chunks = self.text_splitter.split_documents(document)
        
        controls = []
        for chunk in chunks:
            controls.extend(self._parse_control_section(chunk.page_content))
        return {
            "type": "txt",
            "content": controls
        }
    
    def _analyze_page_layout(self, text: str) -> Dict[str, any]:
        """Analyze the layout structure of text"""
        lines = text.split('\n')
        return {
            "line_count": len(lines),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0,
            "indentation_levels": self._detect_indentation_levels(lines)
        }
    
    def _detect_indentation_levels(self, lines: List[str]) -> Dict[int, int]:
        """Detect and count different indentation levels"""
        indentation_counts = {}
        for line in lines:
            leading_spaces = len(line) - len(line.lstrip())
            if leading_spaces > 0:
                indentation_counts[leading_spaces] = indentation_counts.get(leading_spaces, 0) + 1
        return indentation_counts

    def _parse_control_section(self, text: str) -> List[Dict[str, str]]:
        """Parse a section of text to extract control information"""
        controls = []
        lines = text.split('\n')
        current_control = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a control identifier
            if self._is_control_identifier(line):
                if current_control:
                    controls.append(current_control)
                current_control = {'identifier': line, 'content': ''}
            elif current_control:
                current_control['content'] += f" {line}"
        
        # Add the last control if exists
        if current_control:
            controls.append(current_control)
            
        return controls

    def _is_control_identifier(self, text: str) -> bool:
        """Check if text represents an ISO 27001 control identifier"""
        text = text.strip()
        # Match patterns like "5.1", "A.5.1.1", etc.
        return (
            bool(text) and
            (text[0].isdigit() or text.startswith('A.')) and
            any(char.isdigit() for char in text) and
            len(text) <= 10  # Reasonable length for a control ID
        )