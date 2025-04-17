import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import pandas as pd

class ComplianceVisualizer:
    def create_maturity_radar(self, scores: Dict[str, float]) -> go.Figure:
        """Create a radar chart for maturity scores"""
        categories = list(scores.keys())
        values = list(scores.values())
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='rgb(67, 147, 195)'),
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Maturity Assessment Radar"
        )
        return fig

    def create_maturity_distribution(self, distribution: Dict[str, float]) -> go.Figure:
        """Create a bar chart for maturity level distribution"""
        fig = go.Figure(data=[
            go.Bar(
                x=list(distribution.keys()),
                y=list(distribution.values()),
                marker_color=['#ff9999', '#ffcc99', '#99ff99', '#99ccff', '#cc99ff'],
                text=[f"{v*100:.1f}%" for v in distribution.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Control Maturity Distribution",
            xaxis_title="Maturity Level",
            yaxis_title="Proportion of Controls",
            yaxis=dict(tickformat=".0%")
        )
        return fig

    def create_coverage_heatmap(self, sections: List[str], coverage_data: Dict[str, Any]) -> go.Figure:
        """Create a heatmap showing coverage across sections"""
        coverage_matrix = []
        for section in sections:
            section_data = coverage_data.get(section, {})
            mappings = section_data.get('mappings', [])
            row = []
            for mapping in mappings:
                row.append(mapping.get('coverage_score', 0))
            coverage_matrix.append(row)
        
        # Pad rows to same length
        max_len = max(len(row) for row in coverage_matrix)
        coverage_matrix = [row + [None] * (max_len - len(row)) for row in coverage_matrix]
        
        fig = go.Figure(data=go.Heatmap(
            z=coverage_matrix,
            x=[f"Req {i+1}" for i in range(max_len)],
            y=sections,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            showscale=True
        ))
        
        fig.update_layout(
            title="Coverage Heatmap by Section",
            xaxis_title="Requirements",
            yaxis_title="Sections"
        )
        return fig

    def create_improvement_timeline(self, gaps: List[Dict[str, Any]]) -> go.Figure:
        """Create a Gantt chart for suggested improvements"""
        # Sort gaps by priority
        sorted_gaps = sorted(gaps, key=lambda x: x['priority'], reverse=True)
        
        # Create timeline data
        tasks = []
        current_date = pd.Timestamp.now()
        
        for i, gap in enumerate(sorted_gaps):
            # Estimate duration based on effort
            effort = gap.get('estimated_effort', 'medium')
            duration = 30  # default 30 days
            if effort == 'low':
                duration = 14
            elif effort == 'high':
                duration = 60
            
            tasks.append({
                'Task': f"Gap {i+1}",
                'Start': current_date,
                'Finish': current_date + pd.Timedelta(days=duration),
                'Priority': gap['priority'],
                'Coverage': gap['current_coverage']
            })
        
        df = pd.DataFrame(tasks)
        
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task",
                         color="Priority",
                         title="Improvement Timeline")
        
        fig.update_yaxes(autorange="reversed")
        return fig

    def create_progress_indicators(self, analysis_data: Dict[str, Any]) -> List[go.Figure]:
        """Create progress indicator gauges"""
        indicators = []
        
        # Overall compliance gauge
        total_reqs = sum(section['summary']['total_requirements'] 
                        for section in analysis_data.values())
        total_mapped = sum(section['summary']['mapped_requirements'] 
                          for section in analysis_data.values())
        compliance_rate = (total_mapped / total_reqs) if total_reqs > 0 else 0
        
        indicators.append(go.Figure(go.Indicator(
            mode = "gauge+number",
            value = compliance_rate * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Compliance"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ]
            }
        )))
        
        # Average maturity gauge
        if "maturity_summary" in analysis_data:
            maturity = analysis_data["maturity_summary"]["overall_maturity"]
            indicators.append(go.Figure(go.Indicator(
                mode = "gauge+number",
                value = maturity * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Maturity"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ]
                }
            )))
        
        return indicators

    def create_trend_analysis(self, historical_data: List[Dict[str, Any]]) -> go.Figure:
        """Create trend analysis chart for compliance metrics over time"""
        dates = []
        compliance_rates = []
        maturity_scores = []
        
        for data_point in historical_data:
            dates.append(pd.to_datetime(data_point['timestamp']))
            compliance_rates.append(data_point['compliance_rate'])
            maturity_scores.append(data_point.get('maturity_score', None))
        
        fig = go.Figure()
        
        # Add compliance rate line
        fig.add_trace(go.Scatter(
            x=dates,
            y=compliance_rates,
            name="Compliance Rate",
            line=dict(color='blue')
        ))
        
        # Add maturity score line if available
        if any(score is not None for score in maturity_scores):
            fig.add_trace(go.Scatter(
                x=dates,
                y=maturity_scores,
                name="Maturity Score",
                line=dict(color='green')
            ))
        
        fig.update_layout(
            title="Compliance and Maturity Trends",
            xaxis_title="Date",
            yaxis_title="Score",
            yaxis=dict(tickformat=".0%"),
            hovermode='x unified'
        )
        
        return fig