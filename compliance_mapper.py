import json
import os
from datetime import datetime
from typing import Dict, List
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class ComplianceMapper:
    def __init__(self, collection_name: str = "iso27001_2022"):
        """Initialize the compliance mapper with a specific collection"""
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )

    def extract_requirements(self, section: str, threshold: float = 0.75) -> List[Dict[str, str]]:
        """Extract specific requirements from a section of the compliance document"""
        # Query to find requirements from the section
        query = f"What are the key requirements in {section} of ISO 27001:2022?"
        results = self.db.similarity_search_with_relevance_scores(query, k=5)
        
        requirements = []
        for doc, score in results:
            if score >= threshold:
                requirements.append({
                    "section": section,
                    "content": doc.page_content,
                    "relevance": score,
                    "source": doc.metadata
                })
        
        return requirements

    def map_controls(self, requirement: Dict[str, str], existing_controls: List[Dict[str, str]]) -> Dict[str, any]:
        """Map requirements to existing security controls"""
        matches = []
        
        for control in existing_controls:
            # Extract content based on Document Content type
            content = self._extract_control_content(control)
            
            # Calculate relevance between requirement and control
            query = f"How does this control address the requirement? Control: {content}, Requirement: {requirement['content']}"
            results = self.db.similarity_search_with_relevance_scores(query, k=1)
            
            if results and results[0][1] >= 0.7:  # Relevance threshold
                control_info = {
                    "name": control.get('Name (This field is mandatory.)', ''),
                    "content": content,
                    "type": control.get('Document Type (You need to provide the name of the policy type, you can obtain the name of type from Control Catalogue / Policies / Settings / Document Type.)', ''),
                    "version": control.get('Version (This field is mandatory.)', ''),
                    "status": control.get('Status (Mandatory, set value: 0 for Draft, 1 for Published)', '')
                }
                
                matches.append({
                    "control": control_info,
                    "relevance": results[0][1],
                    "strength": self._assess_control_strength(requirement, control_info)
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

    def _extract_control_content(self, control: Dict[str, str]) -> str:
        """Extract the content from a control based on its Document Content type"""
        doc_content = control.get('Document Content (Mandatory, set one of the following values: 0 for Use Content, 1 for Use Attachments, 2 for Use URL)', '0')
        
        if doc_content == '0':  # Use Content
            return control.get('Content Editor Text (This field is mandatory only if Document Content is set to "Use Content")', '')
        elif doc_content == '2':  # Use URL
            return control.get('URL (This field is mandatory only if Document Content is set to "URL")', '')
        else:
            return control.get('Name (This field is mandatory.)', '')

    def _assess_control_strength(self, requirement: Dict[str, str], control: Dict[str, str]) -> float:
        """Assess the strength of a control in meeting a requirement"""
        # Query to assess control effectiveness
        query = f"How effectively does this control satisfy the requirement? Control: {control['content']}, Requirement: {requirement['content']}"
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        return results[0][1] if results else 0.0

    def _calculate_coverage(self, control_matches: List[Dict[str, any]]) -> float:
        """Calculate the overall coverage score for a requirement"""
        if not control_matches:
            return 0.0
        
        # Weight the coverage based on control strengths and relevance
        total_weight = sum(match["relevance"] * match["strength"] for match in control_matches)
        return total_weight / len(control_matches)

    def _generate_gap_description(self, mapping: Dict[str, any]) -> str:
        """Generate a description of identified gaps"""
        query = f"What aspects of this requirement are not fully covered by the controls? Requirement: {mapping['requirement']['content']}"
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        return results[0][0].page_content if results else mapping['requirement']['content']

    def _suggest_improvements(self, mapping: Dict[str, any]) -> List[str]:
        """Suggest improvements to address identified gaps"""
        query = f"How can we improve coverage for this requirement? Requirement: {mapping['requirement']['content']}"
        results = self.db.similarity_search_with_relevance_scores(query, k=3)
        return [doc.page_content for doc, score in results if score >= 0.7]

    def save_analysis(self, filename: str, analysis_data: Dict[str, any]):
        """Save the analysis results to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)

    def load_analysis(self, filename: str) -> Dict[str, any]:
        """Load previous analysis results"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)