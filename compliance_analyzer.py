from compliance_mapper import ComplianceMapper
from policy_validator import PolicyValidator
import json
from typing import Dict, List
from datetime import datetime

def load_existing_controls(csv_file: str) -> List[Dict[str, str]]:
    """Load existing security controls from the CSV file"""
    validator = PolicyValidator(csv_file)
    validator.load_and_validate()
    
    controls = []
    for policy in validator.policies:
        controls.append({
            "name": policy.get('Name (This field is mandatory.)', ''),
            "content": policy.get('Document Content (Mandatory...)', ''),
            "type": policy.get('Document Type', ''),
            "version": policy.get('Version (This field is mandatory.)', ''),
            "status": policy.get('Status (Mandatory...)', '')
        })
    return controls

def analyze_section(mapper: ComplianceMapper, section: str, controls: List[Dict[str, str]]) -> Dict[str, any]:
    """Analyze a specific section of ISO 27001:2022"""
    # Extract requirements for the section
    requirements = mapper.extract_requirements(section)
    
    # Map requirements to controls
    mappings = []
    for req in requirements:
        mapping = mapper.map_controls(req, controls)
        mappings.append(mapping)
    
    # Identify gaps
    gaps = mapper.identify_gaps(mappings)
    
    return {
        "section": section,
        "timestamp": datetime.now().isoformat(),
        "requirements": requirements,
        "mappings": mappings,
        "gaps": gaps,
        "summary": {
            "total_requirements": len(requirements),
            "mapped_requirements": len([m for m in mappings if m["mapped_controls"]]),
            "gaps_identified": len(gaps)
        }
    }

def main():
    # Initialize the compliance mapper
    mapper = ComplianceMapper()
    
    # Load existing security controls from CSV
    controls = load_existing_controls('security-policies.csv')
    
    # Analyze key sections of ISO 27001:2022
    sections = [
        "4 Context of the organization",
        "5 Leadership",
        "6 Planning",
        "7 Support",
        "8 Operation",
        "9 Performance evaluation",
        "10 Improvement"
    ]
    
    analysis_results = {}
    for section in sections:
        print(f"\nAnalyzing section: {section}")
        results = analyze_section(mapper, section, controls)
        analysis_results[section] = results
        
        # Print summary for this section
        print(f"Found {results['summary']['total_requirements']} requirements")
        print(f"Mapped {results['summary']['mapped_requirements']} to existing controls")
        print(f"Identified {results['summary']['gaps_identified']} gaps")
    
    # Save the complete analysis
    mapper.save_analysis('compliance_analysis.json', analysis_results)
    print("\nFull analysis has been saved to compliance_analysis.json")
    
    # Print overall statistics
    total_reqs = sum(section['summary']['total_requirements'] for section in analysis_results.values())
    total_mapped = sum(section['summary']['mapped_requirements'] for section in analysis_results.values())
    total_gaps = sum(section['summary']['gaps_identified'] for section in analysis_results.values())
    
    print("\nOverall Analysis Summary")
    print("=" * 30)
    print(f"Total Requirements: {total_reqs}")
    print(f"Successfully Mapped: {total_mapped}")
    print(f"Gaps Identified: {total_gaps}")
    print(f"Coverage Rate: {(total_mapped/total_reqs)*100:.1f}%")

if __name__ == "__main__":
    main()