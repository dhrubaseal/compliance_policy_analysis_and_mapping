from typing import Dict, List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import json
import os

load_dotenv()  # Load environment variables from .env file

class ComplianceMapper:
    def __init__(self, collection_name: str = "iso27001_2022"):
        """Initialize the compliance mapper with a specific collection"""
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize ChromaDB with explicit settings
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=True
            )
        )
        
        # Initialize Chroma collection
        self.db = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.control_mappings = {}
        self.requirement_cache = {}

    def extract_requirements(self, section: str, threshold: float = 0.75) -> List[Dict[str, str]]:
        """Extract specific requirements from a section of the compliance document"""
        # Query the vector store for relevant content
        results = self.db.similarity_search_with_relevance_scores(
            f"What are the requirements in section {section} of ISO 27001:2022?",
            k=5
        )
        
        requirements = []
        for doc, score in results:
            if score >= threshold:
                # Parse the content to identify specific requirements
                requirements.append({
                    "section": section,
                    "content": doc.page_content,
                    "relevance": score,
                    "source": doc.metadata
                })
        
        # Cache the results
        self.requirement_cache[section] = requirements
        return requirements

    def map_controls(self, requirement: Dict[str, str], existing_controls: List[Dict[str, str]]) -> Dict[str, any]:
        """Map requirements to existing security controls"""
        # Query to find most relevant controls
        query = f"Which security controls address this requirement: {requirement['content']}"
        matches = []
        
        for control in existing_controls:
            # Calculate relevance between requirement and control
            results = self.db.similarity_search_with_relevance_scores(
                f"How does {control['name']} address: {requirement['content']}",
                k=1
            )
            if results and results[0][1] >= 0.7:  # Relevance threshold
                matches.append({
                    "control": control,
                    "relevance": results[0][1],
                    "strength": self._assess_control_strength(requirement, control)
                })

        return {
            "requirement": requirement,
            "mapped_controls": matches,
            "coverage_score": self._calculate_coverage(matches)
        }

    def identify_gaps(self, mappings: List[Dict[str, any]], threshold: float = 0.8) -> List[Dict[str, str]]:
        """Identify gaps in control coverage"""
        gaps = []
        for mapping in mappings:
            coverage = mapping["coverage_score"]
            if coverage < threshold:
                gaps.append({
                    "requirement": mapping["requirement"],
                    "current_coverage": coverage,
                    "gap_description": self._generate_gap_description(mapping),
                    "suggested_improvements": self._suggest_improvements(mapping)
                })
        return gaps

    def _assess_control_strength(self, requirement: Dict[str, str], control: Dict[str, str]) -> float:
        """Assess the strength of a control in meeting a requirement"""
        # Query the vector store to assess control effectiveness
        query = f"How effectively does this control address the requirement? Control: {control['name']}, Requirement: {requirement['content']}"
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        return results[0][1] if results else 0.0

    def _calculate_coverage(self, control_matches: List[Dict[str, any]]) -> float:
        """Calculate the overall coverage score for a requirement"""
        if not control_matches:
            return 0.0
        
        # Weight the coverage based on control strengths
        weights = [match["relevance"] * match["strength"] for match in control_matches]
        return sum(weights) / len(control_matches)

    def _generate_gap_description(self, mapping: Dict[str, any]) -> str:
        """Generate a description of identified gaps"""
        requirement = mapping["requirement"]["content"]
        coverage = mapping["coverage_score"]
        
        query = f"What aspects of this requirement are not fully addressed by existing controls? Requirement: {requirement}, Current coverage: {coverage}"
        results = self.db.similarity_search(query, k=1)
        
        return results[0].page_content if results else "Gap analysis not available"

    def _suggest_improvements(self, mapping: Dict[str, any]) -> List[str]:
        """Suggest improvements to address identified gaps"""
        requirement = mapping["requirement"]["content"]
        current_controls = [m["control"]["name"] for m in mapping["mapped_controls"]]
        
        query = f"What additional controls or improvements would help address this requirement? Requirement: {requirement}, Current controls: {', '.join(current_controls)}"
        results = self.db.similarity_search(query, k=2)
        
        return [doc.page_content for doc in results]

    def save_analysis(self, filename: str, analysis_data: Dict[str, any]):
        """Save the analysis results to a file"""
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)

    def load_analysis(self, filename: str) -> Dict[str, any]:
        """Load previous analysis results"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)