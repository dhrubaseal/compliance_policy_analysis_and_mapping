import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from compliance_analyzer import analyze_section, load_existing_controls
from compliance_mapper import ComplianceMapper

# Page configuration
st.set_page_config(
    page_title="ISO 27001:2022 Compliance Analysis",
    page_icon="🔒",
    layout="wide"
)

# Main title
st.title("ISO 27001:2022 Compliance Analysis Dashboard")

# File uploader for security policies
uploaded_file = st.file_uploader("Upload Security Policies CSV", type=['csv'])

if uploaded_file is not None:
    # Initialize analysis components
    mapper = ComplianceMapper()
    
    # Load controls from uploaded file
    controls = load_existing_controls(uploaded_file)
    
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
    
    analysis_results = {}
    for section in sections:
        results = analyze_section(mapper, section, controls)
        analysis_results[section] = results
    
    # Save the analysis results
    mapper.save_analysis('compliance_analysis.json', analysis_results)
    
    # Load the analysis data
    with open('compliance_analysis.json', 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
        
    # Sidebar with section selection
    st.sidebar.header("Navigation")
    selected_section = st.sidebar.selectbox(
        "Select Section",
        options=list(analysis_data.keys())
    )

    # Main content area
    section_data = analysis_data[selected_section]

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requirements", section_data['summary']['total_requirements'])
    with col2:
        st.metric("Mapped Requirements", section_data['summary']['mapped_requirements'])
    with col3:
        coverage = (section_data['summary']['mapped_requirements'] / section_data['summary']['total_requirements']) * 100
        st.metric("Coverage Rate", f"{coverage:.1f}%")

    # Requirements and Mappings
    st.header("Requirements and Controls Mapping")
    for i, mapping in enumerate(section_data['mappings'], 1):
        with st.expander(f"Requirement {i}"):
            st.markdown("**Content:**")
            st.write(mapping['requirement']['content'])
            
            st.markdown("**Mapped Controls:**")
            if mapping['mapped_controls']:
                for control in mapping['mapped_controls']:
                    st.write(f"- Relevance: {control['relevance']:.2f}")
                    st.write(f"- Strength: {control['strength']:.2f}")
            
            st.markdown("**Coverage Score:**")
            st.progress(mapping['coverage_score'])
            st.write(f"Score: {mapping['coverage_score']:.2f}")

    # Gaps Analysis
    if section_data['gaps']:
        st.header("Identified Gaps")
        for gap in section_data['gaps']:
            st.warning(gap['gap_description'])
else:
    st.info("Please upload a security policies CSV file to begin the analysis.")