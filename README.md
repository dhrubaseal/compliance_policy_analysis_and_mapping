# ISO 27001:2022 Compliance Analysis Tool

A sophisticated tool for analyzing and mapping organizational security policies against ISO/IEC 27001:2022 requirements. This tool helps organizations assess their compliance status, identify gaps, and track their information security management system (ISMS) implementation progress.

## Features

- **Automated Requirements Analysis**: Extracts and analyzes requirements from ISO 27001:2022 sections
- **Control Mapping**: Maps existing security controls to ISO 27001 requirements
- **Gap Analysis**: Identifies gaps between current controls and standard requirements
- **Coverage Metrics**: Calculates and visualizes compliance coverage rates
- **Interactive Dashboard**: Streamlit-based web interface for easy navigation and analysis
- **PDF Processing**: Handles ISO 27001:2022 standard documentation
- **Vector-based Similarity**: Uses advanced text analysis for accurate control mapping

## Project Structure

```
├── app.py                         # Streamlit web application
├── compliance_analyzer.py         # Core analysis engine
├── compliance_mapper.py           # Control mapping logic
├── policy_validator.py           # Policy validation utilities
├── pdf_to_embeddings.py          # PDF processing module
├── requirements.txt              # Project dependencies
├── chroma_db/                    # Vector database storage
└── input files/                  # Input file directory
    └── security-policies.csv     # Security policies file
```

## Requirements

- Python 3.8+
- Streamlit
- ChromaDB
- PyPDF2
- pandas
- plotly

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dhrubaseal/compliance_policy_analysis_and_mapping.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your security policies in CSV format in the `input files` directory
2. Run the Streamlit application:
```bash
streamlit run app.py
```
3. Upload your security policies CSV file through the web interface
4. Navigate through different ISO 27001:2022 sections to view:
   - Requirement mappings
   - Control coverage
   - Gap analysis
   - Compliance metrics

## Analysis Components

The tool analyzes the following ISO 27001:2022 sections:
- Context of the organization (Section 4)
- Leadership (Section 5)
- Planning (Section 6)
- Support (Section 7)
- Operation (Section 8)
- Performance evaluation (Section 9)
- Improvement (Section 10)

## Key Metrics

- **Total Requirements**: Number of requirements per section
- **Mapped Requirements**: Successfully mapped controls
- **Coverage Rate**: Percentage of requirements covered by existing controls
- **Gap Identification**: Areas requiring additional controls

## File Formats

### Security Policies CSV
Your security policies CSV should contain the following columns:
- Policy identifiers
- Policy content
- Control descriptions
- Implementation status

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Add appropriate license information]

## Authors

[Add author information]

## Acknowledgments

- Based on ISO/IEC 27001:2022 standard
- Uses ChromaDB for vector similarity search
- Built with Streamlit for web interface