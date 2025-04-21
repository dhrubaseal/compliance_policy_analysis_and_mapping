import os
os.environ['TORCH_DISABLE_CUSTOM_CLASS_PATHS'] = '1'

import streamlit as st
import json
import pandas as pd
from datetime import datetime
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from compliance_analyzer import analyze_section, load_existing_controls
from compliance_mapper import ComplianceMapper
from maturity_assessor import MaturityLevel
from document_processor import DocumentProcessor
from config import (
    ISO_SECTIONS,
    COVERAGE_THRESHOLD,
    MATURITY_IMPROVEMENT_THRESHOLD,
    DocumentContent,
    DocumentStatus
)

# Page configuration
st.set_page_config(
    page_title="ISO 27001:2022 Compliance Validation",
    page_icon="🔒",
    layout="wide"
)

# Main title and description
st.title("ISO 27001:2022 Compliance Analysis & Mapping")
st.markdown("""
This tool analyzes your security policies and maps them to ISO 27001:2022 requirements. 
It helps identify gaps in coverage and provides recommendations for improvement.
""")

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis View",
    ["Upload & Validate", "Requirements Analysis", "Control Mapping", "Gap Analysis", "Maturity Assessment"]
)

if page == "Upload & Validate":
    st.header("Upload Security Policies")
    uploaded_file = st.file_uploader("Upload Security Policies", type=['csv', 'pdf', 'doc', 'docx', 'txt'])
    
    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            try:
                # Initialize mapper
                mapper = ComplianceMapper()
                
                # Load and validate controls
                controls = load_existing_controls(temp_path)
                
                if isinstance(controls, list) and controls:
                    st.session_state['controls'] = controls
                    st.success(f"Successfully loaded {len(controls)} controls")
                    
                    # Show control summary
                    st.subheader("Control Summary")
                    control_types = pd.DataFrame(controls).get('Control Category (Must match an ISO 27001:2022 control category)', []).value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            px.pie(
                                values=control_types.values,
                                names=control_types.index,
                                title="Distribution of Controls by Category"
                            )
                        )
                    
                    with col2:
                        st.dataframe(pd.DataFrame({
                            'Category': control_types.index,
                            'Count': control_types.values
                        }))
                
                # Display validation errors if any
                if hasattr(controls, 'errors') and controls.errors:
                    st.error("⚠️ Validation Errors Found")
                    st.markdown("### Policy Validation Errors")
                    for error in controls.errors:
                        with st.expander(f"Error in Row {error['row']} - {error['policy_name']}"):
                            for err in error['errors']:
                                st.error(err['error'] if isinstance(err, dict) else err)
                                if isinstance(err, dict) and 'suggestion' in err:
                                    st.info(f"💡 Suggestion: {err['suggestion']}")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                
            finally:
                os.unlink(temp_path)
    else:
        st.info("Please upload a security policies file to begin the validation.")

elif page == "Requirements Analysis":
    if 'controls' not in st.session_state:
        st.warning("Please upload and validate your security policies first.")
    else:
        st.header("Requirements Analysis")
        
        # Section selector
        selected_section = st.selectbox("Select ISO 27001:2022 Section", ISO_SECTIONS)
        
        with st.spinner(f"Analyzing section {selected_section}..."):
            # Analyze selected section
            mapper = ComplianceMapper()
            section_analysis = analyze_section(mapper, selected_section, st.session_state['controls'])
            
            # Display results
            st.subheader("Requirements Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Requirements", section_analysis['summary']['total_requirements'])
            with col2:
                st.metric("Mapped Requirements", section_analysis['summary']['mapped_requirements'])
            with col3:
                coverage = section_analysis['summary']['mapped_requirements'] / max(1, section_analysis['summary']['total_requirements'])
                st.metric("Coverage Rate", f"{coverage:.1%}")
            
            # Show detailed requirements
            st.subheader("Detailed Requirements")
            for i, req in enumerate(section_analysis.get('requirements', [])):
                with st.expander(f"Requirement {i+1}"):
                    st.markdown(f"**Content:** {req['content']}")
                    st.markdown(f"**Confidence Score:** {req.get('confidence_scores', {}).get('overall', 0):.2f}")
                    if 'validation' in req:
                        st.markdown("**Validation:**")
                        st.json(req['validation'])

elif page == "Control Mapping":
    if 'controls' not in st.session_state:
        st.warning("Please upload and validate your security policies first.")
    else:
        st.header("Control Mapping Analysis")
        
        # Section selector
        selected_section = st.selectbox("Select ISO 27001:2022 Section", ISO_SECTIONS)
        
        with st.spinner(f"Analyzing mappings for {selected_section}..."):
            # Get mappings
            mapper = ComplianceMapper()
            section_analysis = analyze_section(mapper, selected_section, st.session_state['controls'])
            
            # Display mapping results
            if section_analysis.get('mappings'):
                for i, mapping in enumerate(section_analysis['mappings']):
                    with st.expander(f"Requirement {i+1} Mapping"):
                        st.markdown(f"**Requirement:** {mapping['requirement']['content']}")
                        st.markdown("**Mapped Controls:**")
                        
                        if mapping.get('mapped_controls'):
                            for j, control in enumerate(mapping['mapped_controls']):
                                st.markdown(f"*Control {j+1}:* {control['control']['name']}")
                                cols = st.columns(3)
                                cols[0].metric("Relevance", f"{control['relevance']:.2f}")
                                cols[1].metric("Strength", f"{control['strength']:.2f}")
                                cols[2].metric("Score", f"{control['weighted_score']:.2f}")
                        else:
                            st.warning("No controls mapped to this requirement")

elif page == "Gap Analysis":
    if 'controls' not in st.session_state:
        st.warning("Please upload and validate your security policies first.")
    else:
        st.header("Gap Analysis")
        
        # Section selector
        selected_section = st.selectbox("Select ISO 27001:2022 Section", ISO_SECTIONS)
        
        with st.spinner(f"Analyzing gaps in {selected_section}..."):
            # Get gap analysis
            mapper = ComplianceMapper()
            section_analysis = analyze_section(mapper, selected_section, st.session_state['controls'])
            
            if section_analysis.get('gaps'):
                # Show gap summary
                st.subheader("Gap Summary")
                gaps_df = pd.DataFrame(section_analysis['gaps'])
                
                fig = px.scatter(
                    gaps_df,
                    x='current_coverage',
                    y='priority',
                    color='estimated_effort',
                    hover_data=['requirement.content'],
                    title="Gap Analysis Matrix"
                )
                st.plotly_chart(fig)
                
                # Show detailed gaps
                st.subheader("Detailed Gap Analysis")
                for gap in section_analysis['gaps']:
                    with st.expander(f"Gap - Priority: {gap['priority']:.2f}"):
                        st.markdown(f"**Requirement:** {gap['requirement']['content']}")
                        st.markdown(f"**Current Coverage:** {gap['current_coverage']:.1%}")
                        st.markdown(f"**Effort Estimate:** {gap['estimated_effort']}")
                        
                        if gap.get('suggested_improvements'):
                            st.markdown("**Suggested Improvements:**")
                            for suggestion in gap['suggested_improvements']:
                                st.markdown(f"- {suggestion}")

elif page == "Maturity Assessment":
    if 'controls' not in st.session_state:
        st.warning("Please upload and validate your security policies first.")
    else:
        st.header("Maturity Assessment")
        
        with st.spinner("Analyzing control maturity..."):
            # Get maturity assessment
            mapper = ComplianceMapper()
            all_controls = st.session_state['controls']
            
            maturity_results = []
            for control in all_controls:
                assessment = mapper.maturity_assessor.assess_control_maturity(control)
                maturity_results.append(assessment)
            
            # Generate maturity report
            maturity_report = mapper.maturity_assessor.generate_maturity_report(maturity_results)
            
            # Display maturity distribution
            st.subheader("Maturity Distribution")
            
            # Create maturity radar chart
            categories = list(maturity_report['average_scores'].keys())
            values = list(maturity_report['average_scores'].values())
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself'
            ))
            fig.update_layout(title="Maturity Assessment Radar")
            st.plotly_chart(fig)
            
            # Show improvement recommendations
            st.subheader("Improvement Recommendations")
            for area in maturity_report.get('improvement_areas', []):
                with st.expander(f"Improve {area['category']}"):
                    st.markdown(f"**Current Score:** {area['current_score']:.2f}")
                    st.markdown(f"**Target Score:** {area['target_score']:.2f}")
                    st.markdown("**Affected Controls:**")
                    for control_id in area['affected_controls']:
                        st.markdown(f"- {control_id}")

# Add a footer with timestamp
st.sidebar.markdown("---")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")