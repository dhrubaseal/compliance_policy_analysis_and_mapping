from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime

class MaturityLevel(Enum):
    INITIAL = 1       # Ad-hoc processes
    DEVELOPING = 2    # Processes are planned but not fully implemented
    DEFINED = 3       # Processes are documented and standardized
    MANAGED = 4       # Processes are monitored and measured
    OPTIMIZING = 5    # Focus on continuous improvement

class MaturityAssessor:
    def __init__(self):
        self.assessment_criteria = {
            "documentation": {
                "weight": 0.25,
                "metrics": ["completeness", "accuracy", "accessibility"]
            },
            "implementation": {
                "weight": 0.35,
                "metrics": ["coverage", "effectiveness", "consistency"]
            },
            "monitoring": {
                "weight": 0.20,
                "metrics": ["measurement", "reporting", "review_frequency"]
            },
            "improvement": {
                "weight": 0.20,
                "metrics": ["feedback_loop", "adaptation", "innovation"]
            }
        }

    def assess_control_maturity(self, control: Dict[str, any]) -> Dict[str, any]:
        """Assess the maturity level of a single control"""
        scores = {}
        
        # Assess documentation
        doc_score = self._assess_documentation(control)
        scores["documentation"] = doc_score * self.assessment_criteria["documentation"]["weight"]
        
        # Assess implementation
        impl_score = self._assess_implementation(control)
        scores["implementation"] = impl_score * self.assessment_criteria["implementation"]["weight"]
        
        # Assess monitoring
        mon_score = self._assess_monitoring(control)
        scores["monitoring"] = mon_score * self.assessment_criteria["monitoring"]["weight"]
        
        # Assess improvement process
        imp_score = self._assess_improvement(control)
        scores["improvement"] = imp_score * self.assessment_criteria["improvement"]["weight"]
        
        # Calculate overall maturity level
        total_score = sum(scores.values())
        maturity_level = self._calculate_maturity_level(total_score)
        
        return {
            "control_id": control.get("identifier", "Unknown"),
            "scores": scores,
            "overall_score": total_score,
            "maturity_level": maturity_level,
            "assessment_date": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations(scores, maturity_level)
        }

    def _assess_documentation(self, control: Dict[str, any]) -> float:
        """Assess documentation quality"""
        score = 0.0
        content = control.get("content", "")
        
        # Check completeness
        if len(content.split()) >= 50:  # Basic length check
            score += 0.3
        
        # Check for key documentation elements
        key_elements = ["purpose", "scope", "requirements", "responsibilities"]
        for element in key_elements:
            if element.lower() in content.lower():
                score += 0.1
        
        # Check for metadata completeness
        if all(key in control for key in ["version", "status", "reviewer_contact"]):
            score += 0.3
            
        return min(score, 1.0)

    def _assess_implementation(self, control: Dict[str, any]) -> float:
        """Assess implementation effectiveness"""
        score = 0.0
        
        # Check implementation status
        status = control.get("status", "")
        if status == "1":  # Published status
            score += 0.4
        
        # Check for implementation evidence
        if control.get("Document Content") == "0":  # Has actual content
            score += 0.3
            
        # Check for implementation roles
        if control.get("grc_contact") and control.get("reviewer_contact"):
            score += 0.3
            
        return min(score, 1.0)

    def _assess_monitoring(self, control: Dict[str, any]) -> float:
        """Assess monitoring and measurement"""
        score = 0.0
        content = control.get("content", "").lower()
        
        # Check for monitoring indicators
        monitoring_terms = ["monitor", "measure", "review", "audit", "assess"]
        for term in monitoring_terms:
            if term in content:
                score += 0.2
        
        # Check review date
        if control.get("Next Review Date"):
            score += 0.4
            
        # Check for measurement criteria
        measurement_terms = ["metric", "kpi", "indicator", "threshold"]
        for term in measurement_terms:
            if term in content:
                score += 0.1
                
        return min(score, 1.0)

    def _assess_improvement(self, control: Dict[str, any]) -> float:
        """Assess continuous improvement aspects"""
        score = 0.0
        content = control.get("content", "").lower()
        
        # Check for improvement indicators
        improvement_terms = ["improve", "enhance", "optimize", "update", "revise"]
        for term in improvement_terms:
            if term in content:
                score += 0.2
        
        # Check version history
        if control.get("version", "0") > "1":
            score += 0.3
            
        # Check feedback mechanisms
        feedback_terms = ["feedback", "review", "suggestion", "input"]
        for term in feedback_terms:
            if term in content:
                score += 0.1
                
        return min(score, 1.0)

    def _calculate_maturity_level(self, total_score: float) -> MaturityLevel:
        """Calculate maturity level based on total score"""
        if total_score < 0.2:
            return MaturityLevel.INITIAL
        elif total_score < 0.4:
            return MaturityLevel.DEVELOPING
        elif total_score < 0.6:
            return MaturityLevel.DEFINED
        elif total_score < 0.8:
            return MaturityLevel.MANAGED
        else:
            return MaturityLevel.OPTIMIZING

    def _generate_recommendations(self, scores: Dict[str, float], maturity_level: MaturityLevel) -> List[str]:
        """Generate improvement recommendations based on scores"""
        recommendations = []
        
        # Documentation recommendations
        if scores["documentation"] < 0.6:
            recommendations.append("Enhance documentation completeness and ensure all key elements are covered")
        
        # Implementation recommendations
        if scores["implementation"] < 0.7:
            recommendations.append("Strengthen implementation by ensuring consistent application of controls")
        
        # Monitoring recommendations
        if scores["monitoring"] < 0.5:
            recommendations.append("Implement regular monitoring and measurement processes")
        
        # Improvement recommendations
        if scores["improvement"] < 0.4:
            recommendations.append("Establish feedback loops and continuous improvement mechanisms")
        
        # Level-specific recommendations
        if maturity_level == MaturityLevel.INITIAL:
            recommendations.append("Focus on documenting and standardizing basic processes")
        elif maturity_level == MaturityLevel.DEVELOPING:
            recommendations.append("Work on implementing consistent control measures across the organization")
        elif maturity_level == MaturityLevel.DEFINED:
            recommendations.append("Enhance monitoring and measurement capabilities")
        elif maturity_level == MaturityLevel.MANAGED:
            recommendations.append("Focus on optimization and automation opportunities")
        
        return recommendations

    def generate_maturity_report(self, assessments: List[Dict[str, any]]) -> Dict[str, any]:
        """Generate a comprehensive maturity assessment report"""
        total_controls = len(assessments)
        maturity_distribution = {level: 0 for level in MaturityLevel}
        
        for assessment in assessments:
            maturity_distribution[assessment["maturity_level"]] += 1
        
        return {
            "report_date": datetime.now().isoformat(),
            "total_controls": total_controls,
            "maturity_distribution": {
                level.name: count/total_controls for level, count in maturity_distribution.items()
            },
            "average_scores": {
                category: sum(a["scores"][category] for a in assessments) / total_controls
                for category in self.assessment_criteria.keys()
            },
            "overall_maturity": sum(a["overall_score"] for a in assessments) / total_controls,
            "improvement_areas": self._identify_improvement_areas(assessments)
        }

    def _identify_improvement_areas(self, assessments: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Identify key areas needing improvement"""
        improvement_areas = []
        
        # Analyze each assessment criteria
        for category in self.assessment_criteria.keys():
            avg_score = sum(a["scores"][category] for a in assessments) / len(assessments)
            if avg_score < 0.6:  # Threshold for identifying improvement areas
                improvement_areas.append({
                    "category": category,
                    "current_score": avg_score,
                    "target_score": min(avg_score + 0.2, 1.0),  # Incremental improvement target
                    "affected_controls": [
                        a["control_id"] for a in assessments 
                        if a["scores"][category] < 0.6
                    ]
                })
        
        return improvement_areas