import os
os.environ['TORCH_DISABLE_CUSTOM_CLASS_PATHS'] = '1'

import json
from datetime import datetime
from typing import Dict, List, Tuple
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from maturity_assessor import MaturityAssessor, MaturityLevel
from dotenv import load_dotenv
from rapidfuzz import fuzz

# Lazy load PyTorch-dependent imports
sentence_transformer = None
def get_sentence_transformer():
    global sentence_transformer
    if sentence_transformer is None:
        from sentence_transformers import SentenceTransformer
        sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_transformer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    MINIMUM_RELEVANCE_SCORE,
    MAX_SEARCH_RESULTS,
    COVERAGE_THRESHOLD,
    MATURITY_IMPROVEMENT_THRESHOLD,
    DocumentContent,
    DocumentStatus
)

# Load environment variables
load_dotenv()

class ComplianceMapper:
    def __init__(self, collection_name: str = COLLECTION_NAME, api_key: str = None):
        """Initialize the compliance mapper with enhanced NLP capabilities"""
        self.collection_name = collection_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided")
            
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        self.maturity_assessor = MaturityAssessor()
        
        # Initialize sentence transformer lazily
        self.sentence_model = None
        
        # Define requirement patterns
        self.requirement_patterns = {
            'mandatory': {
                'indicators': ['shall', 'must', 'required', 'mandatory'],
                'weight': 1.0
            },
            'recommended': {
                'indicators': ['should', 'recommended', 'may'],
                'weight': 0.8
            },
            'measurement': {
                'indicators': ['measure', 'assess', 'evaluate', 'monitor'],
                'weight': 0.7
            },
            'validation': {
                'indicators': ['verify', 'validate', 'test', 'audit'],
                'weight': 0.7
            }
        }

    def extract_requirements(self, section: str, threshold: float = MINIMUM_RELEVANCE_SCORE) -> List[Dict[str, any]]:
        """Extract requirements with enhanced context awareness"""
        # First, get section context
        context_query = f"What is the main objective and scope of {section} in ISO 27001:2022?"
        context_results = self.db.similarity_search_with_relevance_scores(context_query, k=1)
        section_context = context_results[0][0].page_content if context_results else ""
        
        # Get requirements with context
        requirements_query = f"""
        Given the context: {section_context}
        Extract specific requirements from {section} of ISO 27001:2022, including:
        1. Mandatory requirements (shall/must statements)
        2. Recommended practices (should statements)
        3. Implementation criteria
        4. Measurement requirements
        """
        
        results = self.db.similarity_search_with_relevance_scores(requirements_query, k=MAX_SEARCH_RESULTS)
        
        requirements = []
        seen_contents = set()
        
        for doc, score in results:
            if score >= threshold and doc.page_content not in seen_contents:
                # Analyze requirement with context
                requirement = self._analyze_requirement_with_context(
                    doc.page_content,
                    section_context,
                    section
                )
                
                if requirement['confidence_score'] >= 0.8:
                    requirements.append(requirement)
                    seen_contents.add(doc.page_content)
        
        # Sort by confidence and relevance
        requirements.sort(key=lambda x: x['confidence_score'], reverse=True)
        return requirements

    def _analyze_requirement_with_context(
        self, 
        content: str, 
        context: str, 
        section: str
    ) -> Dict[str, any]:
        """Analyze requirement with context awareness"""
        # Calculate contextual relevance
        context_similarity = self._calculate_semantic_similarity(content, context)
        
        # Analyze requirement characteristics
        requirement_info = {
            'content': content,
            'section': section,
            'context_relevance': context_similarity,
            'type': self._determine_requirement_type(content),
            'indicators': self._extract_requirement_indicators(content),
            'implementation_criteria': self._extract_implementation_criteria(content),
            'measurement_criteria': self._extract_measurement_criteria(content),
            'dependencies': self._extract_dependencies(content, context)
        }
        
        # Calculate confidence score
        requirement_info['confidence_score'] = self._calculate_requirement_confidence(requirement_info)
        
        return requirement_info

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.sentence_model is None:
            self.sentence_model = get_sentence_transformer()
            
        # Generate embeddings
        emb1 = self.sentence_model.encode([text1])[0]
        emb2 = self.sentence_model.encode([text2])[0]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)

    def _determine_requirement_type(self, content: str) -> str:
        """Determine requirement type based on indicators and context"""
        content_lower = content.lower()
        type_scores = {}
        
        for req_type, pattern_info in self.requirement_patterns.items():
            score = sum(
                pattern_info['weight'] 
                for indicator in pattern_info['indicators']
                if indicator in content_lower
            )
            type_scores[req_type] = score
            
        if not any(type_scores.values()):
            return 'informative'
            
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _extract_requirement_indicators(self, content: str) -> List[Dict[str, any]]:
        """Extract and categorize requirement indicators"""
        content_lower = content.lower()
        indicators = []
        
        for category, pattern_info in self.requirement_patterns.items():
            found_indicators = [
                {
                    'term': indicator,
                    'position': content_lower.index(indicator),
                    'category': category
                }
                for indicator in pattern_info['indicators']
                if indicator in content_lower
            ]
            indicators.extend(found_indicators)
            
        return sorted(indicators, key=lambda x: x['position'])

    def _extract_implementation_criteria(self, content: str) -> List[str]:
        """Extract implementation criteria from requirement"""
        impl_indicators = [
            "by", "through", "using", "via", "by means of",
            "implement", "establish", "maintain", "ensure"
        ]
        
        criteria = []
        sentences = content.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in impl_indicators):
                criteria.append(sentence.strip())
                
        return criteria

    def _extract_measurement_criteria(self, content: str) -> List[str]:
        """Extract measurement and validation criteria"""
        measure_indicators = [
            "measure", "assess", "evaluate", "monitor",
            "verify", "validate", "test", "review",
            "audit", "check", "inspect"
        ]
        
        criteria = []
        sentences = content.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in measure_indicators):
                criteria.append(sentence.strip())
                
        return criteria

    def _extract_dependencies(self, content: str, context: str) -> List[str]:
        """Extract dependencies and relationships between requirements"""
        dependency_indicators = [
            "depends on", "requires", "prerequisite",
            "before", "after", "based on", "according to"
        ]
        
        dependencies = []
        combined_text = f"{content} {context}"
        sentences = combined_text.split('.')
        
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in dependency_indicators):
                dependencies.append(sentence.strip())
                
        return dependencies

    def _calculate_requirement_confidence(self, req_info: Dict[str, any]) -> float:
        """Calculate confidence score for requirement identification"""
        scores = []
        
        # Score based on requirement type and indicators
        if req_info['type'] != 'informative':
            type_weight = self.requirement_patterns[req_info['type']]['weight']
            scores.append(type_weight)
        
        # Score based on implementation criteria
        if req_info['implementation_criteria']:
            scores.append(min(1.0, len(req_info['implementation_criteria']) * 0.3))
        
        # Score based on measurement criteria
        if req_info['measurement_criteria']:
            scores.append(min(1.0, len(req_info['measurement_criteria']) * 0.3))
        
        # Score based on context relevance
        scores.append(req_info['context_relevance'])
        
        # Calculate weighted average
        weights = [0.4, 0.2, 0.2, 0.2]  # Weights for type, implementation, measurement, context
        return sum(score * weight for score, weight in zip(scores, weights))

    def map_controls(self, requirement: Dict[str, any], existing_controls: List[Dict[str, str]]) -> Dict[str, any]:
        """Map requirements to controls with enhanced semantic matching"""
        matches = []
        requirement_embedding = self.sentence_model.encode([requirement['content']])[0]
        
        for control in existing_controls:
            # Extract control content
            content = self._extract_control_content(control)
            
            # Calculate semantic similarity
            content_embedding = self.sentence_model.encode([content])[0]
            semantic_similarity = float(cosine_similarity([requirement_embedding], [content_embedding])[0][0])
            
            if semantic_similarity >= MINIMUM_RELEVANCE_SCORE:
                # Create control info with enhanced metadata
                control_info = {
                    'name': control.get('Name (This field is mandatory.)', ''),
                    'content': content,
                    'type': control.get('Document Type', ''),
                    'version': control.get('Version (This field is mandatory.)', ''),
                    'status': control.get('Status (Mandatory, set value: 0 for Draft, 1 for Published)', ''),
                    'permissions': control.get('Policy Portal Permissions for this Document', ''),
                    'metadata': {
                        'last_review': control.get('Last Review Date', ''),
                        'next_review': control.get('Next Review Date', ''),
                        'owner': control.get('GRC Contact', ''),
                        'reviewer': control.get('Policy Reviewer Contact', '')
                    }
                }
                
                # Calculate control effectiveness
                control_assessment = self._assess_control_effectiveness(
                    requirement,
                    control_info,
                    semantic_similarity
                )
                
                # Get maturity assessment
                maturity_assessment = self.maturity_assessor.assess_control_maturity(control_info)
                
                matches.append({
                    'control': control_info,
                    'assessment': control_assessment,
                    'maturity': maturity_assessment,
                    'semantic_similarity': semantic_similarity
                })
        
        # Sort matches by weighted score
        matches.sort(key=lambda x: x['assessment']['weighted_score'], reverse=True)
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(requirement, matches)
        
        return {
            'requirement': requirement,
            'matches': matches[:5],  # Top 5 matches
            'coverage_metrics': coverage_metrics,
            'timestamp': datetime.now().isoformat()
        }

    def _assess_control_effectiveness(
        self,
        requirement: Dict[str, any],
        control: Dict[str, any],
        semantic_similarity: float
    ) -> Dict[str, any]:
        """Assess control effectiveness against requirement"""
        # Base score from semantic similarity
        effectiveness_score = semantic_similarity
        
        # Adjust based on control completeness
        if control['status'] == DocumentStatus.PUBLISHED:
            effectiveness_score += 0.1
        
        if all(key in control['metadata'] for key in ['owner', 'reviewer']):
            effectiveness_score += 0.1
            
        # Adjust based on requirement coverage
        if requirement['type'] == 'mandatory':
            coverage_weight = 1.0
        else:
            coverage_weight = 0.8
            
        weighted_score = effectiveness_score * coverage_weight
        
        return {
            'effectiveness_score': min(effectiveness_score, 1.0),
            'coverage_weight': coverage_weight,
            'weighted_score': min(weighted_score, 1.0)
        }

    def _calculate_coverage_metrics(
        self,
        requirement: Dict[str, any],
        matches: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Calculate detailed coverage metrics"""
        if not matches:
            return {
                'overall_coverage': 0.0,
                'effectiveness_coverage': 0.0,
                'implementation_coverage': 0.0,
                'gaps': ['No controls mapped to this requirement']
            }
            
        # Calculate different aspects of coverage
        effectiveness_scores = [m['assessment']['effectiveness_score'] for m in matches]
        semantic_scores = [m['semantic_similarity'] for m in matches]
        
        # Weight the scores based on requirement type
        weight = 1.0 if requirement['type'] == 'mandatory' else 0.8
        
        metrics = {
            'overall_coverage': min(1.0, sum(semantic_scores) / len(matches) * weight),
            'effectiveness_coverage': min(1.0, sum(effectiveness_scores) / len(matches) * weight),
            'implementation_coverage': min(1.0, len(matches) / MAX_SEARCH_RESULTS * weight)
        }
        
        # Identify gaps
        gaps = []
        if metrics['overall_coverage'] < COVERAGE_THRESHOLD:
            gaps.append('Insufficient overall coverage')
        if metrics['effectiveness_coverage'] < COVERAGE_THRESHOLD:
            gaps.append('Controls not fully effective')
        if metrics['implementation_coverage'] < COVERAGE_THRESHOLD:
            gaps.append('Implementation coverage needs improvement')
            
        metrics['gaps'] = gaps
        return metrics

    def _extract_control_content(self, control: Dict[str, str]) -> str:
        """Extract control content based on document type"""
        if control.get('Document Content') == DocumentContent.USE_CONTENT:
            return control.get('Content Editor Text', '')
        elif control.get('Document Content') == DocumentContent.USE_ATTACHMENTS:
            return control.get('Attachments', '')
        elif control.get('Document Content') == DocumentContent.USE_URL:
            return control.get('URL', '')
        return ''

    def identify_gaps(self, mappings: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Identify gaps in control coverage and generate recommendations"""
        gaps = []
        
        for mapping in mappings:
            requirement = mapping['requirement']
            coverage_metrics = mapping.get('coverage_metrics', {})
            
            # Check if this requirement has insufficient coverage
            if (coverage_metrics.get('overall_coverage', 0) < COVERAGE_THRESHOLD or 
                coverage_metrics.get('effectiveness_coverage', 0) < COVERAGE_THRESHOLD):
                
                # Calculate gap priority based on requirement type and coverage
                priority = self._calculate_gap_priority(requirement, coverage_metrics)
                
                # Estimate implementation effort
                estimated_effort = self._estimate_implementation_effort(requirement)
                
                # Generate improvement suggestions
                suggestions = self._generate_improvement_suggestions(requirement, coverage_metrics)
                
                gaps.append({
                    'requirement': requirement,
                    'current_coverage': coverage_metrics.get('overall_coverage', 0),
                    'priority': priority,
                    'estimated_effort': estimated_effort,
                    'suggested_improvements': suggestions
                })
        
        # Sort gaps by priority (highest first)
        gaps.sort(key=lambda x: x['priority'], reverse=True)
        return gaps
        
    def _calculate_gap_priority(self, requirement: Dict[str, any], coverage_metrics: Dict[str, float]) -> float:
        """Calculate priority score for a gap"""
        # Base priority on requirement type
        type_weights = {
            'mandatory': 1.0,
            'recommended': 0.8,
            'measurement': 0.7,
            'validation': 0.7,
            'informative': 0.5
        }
        
        base_priority = type_weights.get(requirement['type'], 0.5)
        
        # Adjust based on current coverage
        coverage_factor = 1 - coverage_metrics.get('overall_coverage', 0)
        
        # Consider implementation status
        impl_factor = 1 - coverage_metrics.get('implementation_coverage', 0)
        
        # Calculate final priority score
        priority = (base_priority * 0.4 + coverage_factor * 0.4 + impl_factor * 0.2)
        
        return min(1.0, priority)
        
    def _estimate_implementation_effort(self, requirement: Dict[str, any]) -> str:
        """Estimate the effort required to implement a control"""
        # Count complexity indicators
        complexity_indicators = {
            'high': ['develop', 'implement', 'establish', 'maintain', 'monitor'],
            'medium': ['document', 'define', 'specify', 'review'],
            'low': ['identify', 'list', 'record', 'report']
        }
        
        content = requirement['content'].lower()
        scores = {level: sum(1 for ind in indicators if ind in content)
                 for level, indicators in complexity_indicators.items()}
        
        # Consider dependencies
        if requirement.get('dependencies'):
            scores['high'] += len(requirement['dependencies'])
        
        # Determine effort level
        if scores['high'] > 0:
            return 'High'
        elif scores['medium'] > 0:
            return 'Medium'
        else:
            return 'Low'
        
    def _generate_improvement_suggestions(
        self,
        requirement: Dict[str, any],
        coverage_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate specific improvement suggestions based on gaps"""
        suggestions = []
        
        # Check overall coverage
        if coverage_metrics.get('overall_coverage', 0) < COVERAGE_THRESHOLD:
            suggestions.append(
                f"Develop a new control that specifically addresses: {requirement['content']}"
            )
        
        # Check effectiveness
        if coverage_metrics.get('effectiveness_coverage', 0) < COVERAGE_THRESHOLD:
            suggestions.append(
                "Enhance existing controls with more specific implementation details"
            )
        
        # Add implementation criteria
        if requirement.get('implementation_criteria'):
            suggestions.append(
                f"Ensure implementation includes: {', '.join(requirement['implementation_criteria'])}"
            )
        
        # Add measurement criteria
        if requirement.get('measurement_criteria'):
            suggestions.append(
                f"Add measurement mechanisms: {', '.join(requirement['measurement_criteria'])}"
            )
        
        return suggestions