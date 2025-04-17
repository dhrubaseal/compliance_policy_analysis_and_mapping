import json
import os
from datetime import datetime
from typing import Dict, List
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from maturity_assessor import MaturityAssessor, MaturityLevel
from dotenv import load_dotenv
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

class ComplianceMapper:
    def __init__(self, collection_name: str = "iso27001_2022", api_key: str = None):
        """Initialize the compliance mapper with a specific collection"""
        self.collection_name = collection_name
        
        # Get API key from parameter, environment variable, or raise error
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through api_key parameter or OPENAI_API_KEY environment variable")
            
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"
        )
        self.maturity_assessor = MaturityAssessor()
        self.requirement_patterns = {
            "mandatory": ["shall", "must", "required", "mandatory"],
            "recommended": ["should", "recommended", "may"],
            "measurement": ["measure", "assess", "evaluate", "monitor"],
            "validation": ["verify", "validate", "test", "audit"]
        }

    def validate_requirement(self, requirement: Dict[str, str]) -> Dict[str, any]:
        """Validate and enrich a requirement with additional metadata"""
        # Analyze requirement structure and completeness
        validation_query = f"""Analyze this ISO 27001:2022 requirement and extract:
        1. The requirement type (mandatory/recommended)
        2. The implementation complexity (1-5)
        3. Dependencies on other controls
        4. Documentation needs
        5. Validation criteria

        Requirement: {requirement['content']}
        """
        
        results = self.db.similarity_search_with_relevance_scores(validation_query, k=1)
        if not results:
            return requirement
            
        analysis = results[0][0].page_content
        
        # Extract structured validation data
        validation_data = {
            "type": self._extract_between(analysis, "requirement type:", "implementation complexity:").strip(),
            "complexity": int(self._extract_between(analysis, "implementation complexity:", "dependencies:").strip() or "3"),
            "dependencies": self._extract_between(analysis, "dependencies:", "documentation needs:").strip(),
            "documentation": self._extract_between(analysis, "documentation needs:", "validation criteria:").strip(),
            "validation_criteria": self._extract_between(analysis, "validation criteria:", "\n").strip()
        }
        
        # Analyze requirement quality
        quality_issues = self._check_requirement_quality(requirement['content'])
        
        # Combine original requirement with validation data
        enriched_requirement = {
            **requirement,
            "validation": validation_data,
            "quality_issues": quality_issues,
            "is_valid": len(quality_issues) == 0
        }
        
        return enriched_requirement

    def _check_requirement_quality(self, content: str) -> List[str]:
        """Check requirement content for quality issues"""
        issues = []
        
        # Common quality checks
        if len(content.split()) < 5:
            issues.append("Requirement is too brief")
        if len(content.split()) > 100:
            issues.append("Requirement is too verbose")
        
        # Check for ambiguous language
        ambiguous_terms = [
            "appropriate", "adequate", "sufficient", "reasonable",
            "etc", "and/or", "as needed", "if possible"
        ]
        found_terms = [term for term in ambiguous_terms if term in content.lower()]
        if found_terms:
            issues.append(f"Contains ambiguous terms: {', '.join(found_terms)}")
        
        # Check for measurability
        if not any(term in content.lower() for term in ["must", "shall", "will", "should"]):
            issues.append("No clear mandate or requirement indicator")
        
        # Check for testability
        if not any(term in content.lower() for term in ["verify", "measure", "test", "review", "audit", "assess"]):
            issues.append("No clear validation or verification criteria")
        
        return issues

    def extract_requirements(self, section: str, threshold: float = 0.75) -> List[Dict[str, str]]:
        """Extract and validate requirements with enhanced accuracy"""
        section_query = f"What are the key requirements in {section} of ISO 27001:2022?"
        section_results = self.db.similarity_search_with_relevance_scores(section_query, k=10)
        
        requirements = []
        seen_contents = set()
        
        for doc, score in section_results:
            if score >= threshold and doc.page_content not in seen_contents:
                # Enhanced requirement verification with confidence scoring
                confidence_scores = self._calculate_requirement_confidence(doc.page_content)
                
                if confidence_scores["overall"] >= 0.8:
                    requirement = {
                        "section": section,
                        "content": doc.page_content,
                        "relevance": score,
                        "source": doc.metadata,
                        "confidence_scores": confidence_scores
                    }
                    
                    validated_req = self.validate_requirement(requirement)
                    if validated_req.get("is_valid", False):
                        requirements.append(validated_req)
                        seen_contents.add(doc.page_content)
        
        requirements.sort(key=lambda x: x["relevance"] * x["confidence_scores"]["overall"], reverse=True)
        self._add_section_context(requirements)
        return requirements

    def _calculate_requirement_confidence(self, content: str) -> Dict[str, float]:
        """Calculate confidence scores for requirement identification"""
        content_lower = content.lower()
        scores = {
            "mandate_clarity": 0.0,
            "measurability": 0.0,
            "completeness": 0.0,
            "ambiguity": 0.0
        }
        
        # Check mandate clarity using fuzzy matching
        mandate_matches = []
        for pattern_type, patterns in self.requirement_patterns.items():
            for pattern in patterns:
                highest_ratio = max(
                    fuzz.partial_ratio(pattern, phrase) 
                    for phrase in content_lower.split('.')
                )
                mandate_matches.append(highest_ratio / 100.0)
        scores["mandate_clarity"] = max(mandate_matches) if mandate_matches else 0.0
        
        # Check measurability
        measurable_elements = sum(
            1 for pattern in self.requirement_patterns["measurement"] 
            if pattern in content_lower
        )
        scores["measurability"] = min(1.0, measurable_elements / 2)
        
        # Check completeness (presence of subject, action, and condition)
        has_subject = any(word.lower() in content_lower for word in ["organization", "staff", "personnel", "management"])
        has_action = any(word in content_lower for word in ["implement", "establish", "maintain", "ensure"])
        has_condition = any(word in content_lower for word in ["when", "if", "during", "while"])
        scores["completeness"] = (has_subject + has_action + has_condition) / 3
        
        # Calculate ambiguity score (inverse of clarity)
        ambiguous_phrases = [
            "as appropriate", "if applicable", "as necessary", "etc",
            "and/or", "adequate", "sufficient", "reasonable"
        ]
        ambiguity_matches = sum(1 for phrase in ambiguous_phrases if phrase in content_lower)
        scores["ambiguity"] = max(0, 1 - (ambiguity_matches * 0.2))
        
        # Calculate overall confidence score with weighted components
        scores["overall"] = (
            scores["mandate_clarity"] * 0.4 +
            scores["measurability"] * 0.3 +
            scores["completeness"] * 0.2 +
            scores["ambiguity"] * 0.1
        )
        
        return scores

    def _add_section_context(self, requirements: List[Dict[str, any]]) -> None:
        """Add section context and relationships between requirements"""
        if not requirements:
            return
            
        section = requirements[0]["section"]
        
        # Get section context
        context_query = f"What is the main objective and context of section {section} in ISO 27001:2022?"
        context_results = self.db.similarity_search_with_relevance_scores(context_query, k=1)
        
        if context_results:
            section_context = context_results[0][0].page_content
            
            # Find relationships between requirements
            for i, req in enumerate(requirements):
                # Find related requirements
                related = []
                for j, other_req in enumerate(requirements):
                    if i != j:
                        relation_score = self._calculate_requirement_relationship(req, other_req)
                        if relation_score > 0.7:  # High relationship threshold
                            related.append({
                                "requirement_id": j,
                                "relationship_type": self._determine_relationship_type(req, other_req),
                                "strength": relation_score
                            })
                
                # Add context to requirement
                req["section_context"] = section_context
                req["related_requirements"] = related

    def _calculate_requirement_relationship(self, req1: Dict[str, any], req2: Dict[str, any]) -> float:
        """Calculate the relationship strength between two requirements"""
        query = f"""How strongly are these requirements related?
        Requirement 1: {req1['content']}
        Requirement 2: {req2['content']}
        """
        
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        return results[0][1] if results else 0.0

    def _determine_relationship_type(self, req1: Dict[str, any], req2: Dict[str, any]) -> str:
        """Determine the type of relationship between requirements"""
        query = f"""What is the relationship between these requirements?
        Choose one: prerequisite, dependent, complementary, conflicting
        
        Requirement 1: {req1['content']}
        Requirement 2: {req2['content']}
        """
        
        results = self.db.similarity_search_with_relevance_scores(query, k=1)
        if not results:
            return "complementary"
            
        response = results[0][0].page_content.lower()
        if "prerequisite" in response:
            return "prerequisite"
        elif "dependent" in response:
            return "dependent"
        elif "conflicting" in response:
            return "conflicting"
        else:
            return "complementary"

    def _parse_requirement_details(self, content: str) -> Dict[str, any]:
        """Parse detailed information from a requirement text"""
        # Query to extract specific aspects of the requirement
        details_query = f"""Analyze this requirement and extract:
        1. The main objective
        2. Any specific controls or measures mentioned
        3. Any measurable criteria
        4. Dependencies on other requirements
        5. Level of ambiguity (score 0-1)
        
        Requirement: {content}"""
        
        results = self.db.similarity_search_with_relevance_scores(details_query, k=1)
        if not results:
            return {"ambiguity_score": 1.0}  # Default to high ambiguity if analysis fails
            
        try:
            # Extract structured information from the analysis
            analysis = results[0][0].page_content
            return {
                "objective": self._extract_between(analysis, "main objective:", "controls:").strip(),
                "controls": self._extract_between(analysis, "controls:", "criteria:").strip(),
                "criteria": self._extract_between(analysis, "criteria:", "dependencies:").strip(),
                "dependencies": self._extract_between(analysis, "dependencies:", "ambiguity:").strip(),
                "ambiguity_score": float(self._extract_between(analysis, "ambiguity:", "\n").strip() or "0.5")
            }
        except Exception:
            return {"ambiguity_score": 0.5}  # Default to medium ambiguity on parsing failure

    def _extract_between(self, text: str, start: str, end: str) -> str:
        """Helper method to extract text between two markers"""
        try:
            start_idx = text.lower().index(start.lower()) + len(start)
            end_idx = text.lower().index(end.lower())
            return text[start_idx:end_idx]
        except ValueError:
            return ""

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
                    "status": control.get('Status (Mandatory, set value: 0 for Draft, 1 for Published)', ''),
                    "permissions": control.get('Policy Portal Permissions for this Document (Mandatory, set one of the following values: public for Public (Everyone can see the document), private for Private (Document is not shown on the portal), custom-roles for Only specific groups)', ''),
                    "grc_contact": control.get('GRC Contact (Mandatory. Accepts multiple user logins or group names separated by "|")', ''),
                    "reviewer_contact": control.get('Policy Reviewer Contact (Mandatory. Accepts multiple user logins or group names separated by "|")', '')
                }
                
                # Calculate control strength and combine with relevance
                strength = self._assess_control_strength(requirement, control_info)
                weighted_score = (results[0][1] * 0.6) + (strength * 0.4)  # Weight relevance more than strength
                
                # Assess control maturity
                maturity_assessment = self.maturity_assessor.assess_control_maturity(control_info)
                
                matches.append({
                    "control": control_info,
                    "relevance": results[0][1],
                    "strength": strength,
                    "weighted_score": weighted_score,
                    "maturity_assessment": maturity_assessment
                })

        # Sort matches by weighted score
        matches.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        # Calculate normalized coverage score
        coverage_score = self._calculate_coverage(matches)
        
        # Get maturity report for mapped controls
        maturity_report = None
        if matches:
            maturity_assessments = [m["maturity_assessment"] for m in matches]
            maturity_report = self.maturity_assessor.generate_maturity_report(maturity_assessments)

        return {
            "requirement": requirement,
            "mapped_controls": matches[:5],  # Return top 5 most relevant controls
            "coverage_score": coverage_score,
            "maturity_report": maturity_report,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _calculate_coverage(self, control_matches: List[Dict[str, any]]) -> float:
        """Calculate the overall coverage score for a requirement"""
        # Return 0 coverage if there are no control matches
        if not control_matches:
            return 0.0
        
        # Enhanced coverage calculation including maturity
        total_weight = 0.0
        valid_matches = 0
        
        for match in control_matches:
            # Skip matches without maturity assessment
            if not match.get("maturity_assessment") or "overall_score" not in match["maturity_assessment"]:
                continue
                
            # Combine control relevance, strength, and maturity level
            maturity_factor = match["maturity_assessment"]["overall_score"]
            weighted_score = (
                match.get("relevance", 0) * 0.4 +    # Relevance weight
                match.get("strength", 0) * 0.3 +     # Control strength weight
                maturity_factor * 0.3                # Maturity weight
            )
            total_weight += weighted_score
            valid_matches += 1
            
        # Return 0 if no valid matches were found
        if valid_matches == 0:
            return 0.0
            
        return min(total_weight / valid_matches, 1.0)

    def identify_gaps(self, mappings: List[Dict[str, any]], threshold: float = 0.8) -> List[Dict[str, str]]:
        """Identify gaps in control coverage"""
        gaps = []
        for mapping in mappings:
            coverage = mapping["coverage_score"]
            if coverage < threshold:
                # Get gap analysis with qualitative assessment
                gap_analysis = self._analyze_gap(mapping)
                
                # Get prioritized improvement suggestions
                improvements = self._suggest_improvements(mapping)
                
                # Get maturity-based recommendations
                if mapping.get("maturity_report"):
                    maturity_recommendations = self._get_maturity_recommendations(mapping["maturity_report"])
                else:
                    maturity_recommendations = []
                
                gaps.append({
                    "requirement": mapping["requirement"],
                    "current_coverage": coverage,
                    "gap_analysis": gap_analysis,
                    "suggested_improvements": improvements,
                    "maturity_recommendations": maturity_recommendations,
                    "priority": self._calculate_gap_priority(mapping, coverage),
                    "estimated_effort": self._estimate_remediation_effort(gap_analysis)
                })

        # Sort gaps by priority
        gaps.sort(key=lambda x: x["priority"], reverse=True)
        return gaps

    def _get_maturity_recommendations(self, maturity_report: Dict[str, any]) -> List[str]:
        """Generate recommendations based on maturity assessment"""
        recommendations = []
        
        # Add recommendations based on maturity distribution
        distribution = maturity_report.get("maturity_distribution", {})
        if distribution.get("INITIAL", 0) > 0.3:
            recommendations.append("High proportion of controls at initial maturity level - focus on standardization")
        if distribution.get("DEVELOPING", 0) > 0.4:
            recommendations.append("Many controls still developing - prioritize implementation completion")
            
        # Add recommendations for improvement areas
        for area in maturity_report.get("improvement_areas", []):
            recommendations.append(f"Improve {area['category']} from {area['current_score']:.2f} to target {area['target_score']:.2f}")
            
        return recommendations

    def save_analysis(self, filename: str, analysis_data: Dict[str, any]):
        """Save the analysis results to a file"""
        # Add maturity summary to the analysis data
        all_maturity_assessments = []
        for section in analysis_data.values():
            for mapping in section.get("mappings", []):
                for control_match in mapping.get("mapped_controls", []):
                    if "maturity_assessment" in control_match:
                        all_maturity_assessments.append(control_match["maturity_assessment"])
        
        if all_maturity_assessments:
            analysis_data["maturity_summary"] = self.maturity_assessor.generate_maturity_report(all_maturity_assessments)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2)

    def load_analysis(self, filename: str) -> Dict[str, any]:
        """Load previous analysis results"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)