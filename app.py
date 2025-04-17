import streamlit as st
import json
import pandas as pd
from datetime import datetime
from compliance_analyzer import analyze_section, load_existing_controls
from compliance_mapper import ComplianceMapper
from maturity_assessor import MaturityLevel
from visualization import ComplianceVisualizer

# Page configuration
st.set_page_config(
    page_title="ISO 27001:2022 Compliance Analysis",
    page_icon="🔒",
    layout="wide"
)

# Initialize visualizer
visualizer = ComplianceVisualizer()

# Main title and description
st.title("ISO 27001:2022 Compliance Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of your organization's security policies against ISO 27001:2022 requirements,
including maturity assessment and improvement recommendations.
""")

# File uploader for security policies with multiple format support
uploaded_file = st.file_uploader("Upload Security Policies", type=['csv', 'pdf', 'doc', 'docx', 'txt'])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        # Initialize analysis components
        mapper = ComplianceMapper()
        
        # Load controls from uploaded file
        controls = load_existing_controls(temp_path)
        
        # Analyze key sections
        sections = [
            "4 Context of the organization",
            "5 Leadership",
            "6 Planning",
            "7 Support",
            "8 Operation",
            "9 Performance evaluation",
            "10 Improvement"
        ]
        
        with st.spinner("Analyzing compliance and maturity..."):
            analysis_results = {}
            for section in sections:
                results = analyze_section(mapper, section, controls)
                analysis_results[section] = results
            
            # Save the analysis results
            mapper.save_analysis('compliance_analysis.json', analysis_results)
            
            # Load the analysis data
            with open('compliance_analysis.json', 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
        
        # Create tabs for different views
        overview_tab, details_tab, maturity_tab, gaps_tab = st.tabs([
            "Overview", "Section Details", "Maturity Assessment", "Gap Analysis"
        ])
        
        with overview_tab:
            # Display overall progress indicators
            st.header("Overall Progress")
            indicators = visualizer.create_progress_indicators(analysis_data)
            cols = st.columns(len(indicators))
            for idx, indicator in enumerate(indicators):
                cols[idx].plotly_chart(indicator, use_container_width=True)
            
            # Display coverage heatmap
            st.header("Coverage Analysis")
            heatmap = visualizer.create_coverage_heatmap(sections, analysis_data)
            st.plotly_chart(heatmap, use_container_width=True)
        
        with details_tab:
            # Section selector
            selected_section = st.selectbox(
                "Select Section",
                options=sections
            )
            
            section_data = analysis_data[selected_section]
            
            # Section metrics
            st.header("Section Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Requirements", section_data['summary']['total_requirements'])
            with col2:
                st.metric("Mapped Controls", section_data['summary']['mapped_requirements'])
            with col3:
                coverage = (section_data['summary']['mapped_requirements'] / 
                          section_data['summary']['total_requirements'] * 100)
                st.metric("Coverage", f"{coverage:.1f}%")
            with col4:
                gaps = len(section_data.get('gaps', []))
                st.metric("Gaps Identified", gaps)
            
            # Requirements and mappings
            st.header("Requirements Analysis")
            for i, mapping in enumerate(section_data['mappings'], 1):
                with st.expander(f"Requirement {i}"):
                    st.markdown("**Requirement Content:**")
                    st.write(mapping['requirement']['content'])
                    
                    if mapping['mapped_controls']:
                        st.markdown("**Mapped Controls:**")
                        for control in mapping['mapped_controls']:
                            cols = st.columns([3, 1, 1, 1])
                            cols[0].write(f"Control: {control['control']['name']}")
                            cols[1].metric("Relevance", f"{control['relevance']:.2f}")
                            cols[2].metric("Strength", f"{control['strength']:.2f}")
                            if 'maturity_assessment' in control:
                                cols[3].metric("Maturity", 
                                             control['maturity_assessment']['maturity_level'].name)
                    
                    st.markdown("**Coverage Score:**")
                    st.progress(mapping['coverage_score'])
        
        with maturity_tab:
            st.header("Maturity Assessment")
            
            if "maturity_summary" in analysis_data:
                summary = analysis_data["maturity_summary"]
                
                # Maturity distribution
                st.subheader("Control Maturity Distribution")
                dist_chart = visualizer.create_maturity_distribution(
                    summary["maturity_distribution"]
                )
                st.plotly_chart(dist_chart, use_container_width=True)
                
                # Category scores radar
                st.subheader("Category Scores")
                radar_chart = visualizer.create_maturity_radar(
                    summary["average_scores"]
                )
                st.plotly_chart(radar_chart, use_container_width=True)
                
                # Improvement areas
                if summary.get("improvement_areas"):
                    st.subheader("Improvement Areas")
                    for area in summary["improvement_areas"]:
                        with st.expander(f"Improve {area['category']}"):
                            st.write(f"Current Score: {area['current_score']:.2f}")
                            st.write(f"Target Score: {area['target_score']:.2f}")
                            st.write("Affected Controls:")
                            for control in area['affected_controls']:
                                st.write(f"• {control}")
        
        with gaps_tab:
            st.header("Gap Analysis")
            
            # Collect all gaps
            all_gaps = []
            for section in analysis_data.values():
                if 'gaps' in section:
                    all_gaps.extend(section['gaps'])
            
            if all_gaps:
                # Create improvement timeline
                st.subheader("Improvement Timeline")
                timeline = visualizer.create_improvement_timeline(all_gaps)
                st.plotly_chart(timeline, use_container_width=True)
                
                # Gap details
                st.subheader("Detailed Gap Analysis")
                for gap in sorted(all_gaps, key=lambda x: x['priority'], reverse=True):
                    with st.expander(f"Gap (Priority {gap['priority']}) - Coverage: {gap['current_coverage']:.2f}"):
                        st.write("**Requirement:**")
                        st.write(gap['requirement']['content'])
                        
                        if gap.get('gap_analysis'):
                            st.write("**Analysis:**")
                            for key, value in gap['gap_analysis'].items():
                                st.write(f"- {key}: {value}")
                        
                        if gap.get('maturity_recommendations'):
                            st.write("**Maturity Recommendations:**")
                            for rec in gap['maturity_recommendations']:
                                st.write(f"• {rec}")
                        
                        st.write("**Improvement Actions:**")
                        for suggestion in gap['suggested_improvements']:
                            st.write(f"• {suggestion}")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)
else:
    st.info("Please upload a security policies file to begin the analysis.")
    
    # Show sample visualizations with dummy data
    st.header("Sample Visualizations")
    sample_data = {
        "maturity_distribution": {
            "INITIAL": 0.2,
            "DEVELOPING": 0.3,
            "DEFINED": 0.25,
            "MANAGED": 0.15,
            "OPTIMIZING": 0.1
        },
        "average_scores": {
            "documentation": 0.65,
            "implementation": 0.72,
            "monitoring": 0.58,
            "improvement": 0.45
        }
    }
    
    col1, col2 = st.columns(2)
    with col1:
        dist_chart = visualizer.create_maturity_distribution(sample_data["maturity_distribution"])
        st.plotly_chart(dist_chart, use_container_width=True)
    
    with col2:
        radar_chart = visualizer.create_maturity_radar(sample_data["average_scores"])
        st.plotly_chart(radar_chart, use_container_width=True)