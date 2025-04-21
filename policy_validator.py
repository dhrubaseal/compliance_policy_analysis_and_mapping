import csv
from datetime import datetime
from typing import Dict, List, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    MINIMUM_RELEVANCE_SCORE,
    MAX_SEARCH_RESULTS,
    SUPPORTED_ENCODINGS,
    REQUIRED_HEADERS,
    DocumentStatus,
    DocumentPermissions
)

class PolicyValidator:
    def __init__(self, file_input):
        self.file_input = file_input
        self.policies = []
        self.errors = []
        
        # Load environment variables
        load_dotenv()
        
        # Initialize vector database connection
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
        
        # Get control categories from ISO document
        self.iso_control_categories = self._get_control_categories()
        
        # Get valid assessment values from ISO document
        self.assessment_values = self._get_assessment_values()
        self.valid_risk_levels = self.assessment_values.get('risk_levels', [])
        self.valid_risk_likelihoods = self.assessment_values.get('risk_likelihoods', [])
        self.valid_risk_impacts = self.assessment_values.get('risk_impacts', [])
        self.valid_assessment_statuses = self.assessment_values.get('assessment_statuses', [])
    
    def _get_control_categories(self) -> List[str]:
        """Extract control categories from ISO 27001:2022 document using vector similarity search"""
        query = "List all control categories and their requirements from ISO 27001:2022. Include the full category numbers and names like '5.1 Information security policies' and their subcategories."
        results = self.db.similarity_search_with_relevance_scores(query, k=MAX_SEARCH_RESULTS)
        
        categories = set()
        for doc, score in results:
            if score >= MINIMUM_RELEVANCE_SCORE:
                content = doc.page_content
                lines = content.split('\n')
                for line in lines:
                    if self._is_control_category(line):
                        clean_category = line.strip()
                        # Ensure the category has both number and description
                        if ' ' in clean_category and not clean_category.lower().startswith('a.'):
                            categories.add(clean_category)
        
        return sorted(list(categories))
    
    def _get_assessment_values(self) -> Dict[str, List[str]]:
        """Extract valid assessment values from ISO 27001:2022 document using vector search"""
        queries = {
            'risk_levels': "What are the recommended risk levels or ratings in ISO 27001:2022 risk assessment?",
            'risk_likelihoods': "What are the recommended likelihood levels for risk assessment in ISO 27001:2022?",
            'risk_impacts': "What are the recommended impact levels for risk assessment in ISO 27001:2022?",
            'assessment_statuses': "What are the typical assessment or implementation status categories in ISO 27001:2022?"
        }
        
        values = {
            'risk_levels': [],
            'risk_likelihoods': [],
            'risk_impacts': [],
            'assessment_statuses': []
        }
        
        for key, query in queries.items():
            results = self.db.similarity_search_with_relevance_scores(query, k=3)
            extracted_values = set()
            
            for doc, score in results:
                if score >= MINIMUM_RELEVANCE_SCORE:
                    content = doc.page_content.lower()
                    found_values = self._extract_list_items(content)
                    extracted_values.update(found_values)
            
            # Clean and standardize the extracted values
            clean_values = self._standardize_values(extracted_values, key)
            values[key] = sorted(clean_values) if clean_values else self._get_default_values(key)
                    
        return values
    
    def _standardize_values(self, values: set, category: str) -> List[str]:
        """Standardize and normalize extracted values"""
        standard_values = set()
        
        for value in values:
            # Clean the value
            clean_value = value.strip().title()
            
            # Handle common variations
            if category == 'risk_levels':
                if 'low' in value:
                    standard_values.add('Low')
                elif 'medium' in value or 'moderate' in value:
                    standard_values.add('Medium')
                elif 'high' in value:
                    standard_values.add('High')
                elif 'critical' in value or 'severe' in value:
                    standard_values.add('Critical')
                    
            elif category == 'risk_likelihoods':
                if 'unlikely' in value or 'rare' in value:
                    standard_values.add('Unlikely')
                elif 'possible' in value:
                    standard_values.add('Possible')
                elif 'likely' in value:
                    standard_values.add('Likely')
                elif 'very' in value or 'almost' in value:
                    standard_values.add('Very Likely')
                    
            elif category == 'risk_impacts':
                if 'minor' in value or 'low' in value:
                    standard_values.add('Minor')
                elif 'moderate' in value or 'medium' in value:
                    standard_values.add('Moderate')
                elif 'major' in value or 'high' in value:
                    standard_values.add('Major')
                elif 'severe' in value or 'critical' in value:
                    standard_values.add('Severe')
                    
            elif category == 'assessment_statuses':
                if 'not' in value or 'pending' in value:
                    standard_values.add('Not Started')
                elif 'progress' in value or 'ongoing' in value:
                    standard_values.add('In Progress')
                elif 'complete' in value or 'done' in value:
                    standard_values.add('Completed')
                elif 'review' in value:
                    standard_values.add('Review Required')
        
        return list(standard_values)
    
    def _get_default_values(self, category: str) -> List[str]:
        """Get default values if extraction fails"""
        defaults = {
            'risk_levels': ['Low', 'Medium', 'High', 'Critical'],
            'risk_likelihoods': ['Unlikely', 'Possible', 'Likely', 'Very Likely'],
            'risk_impacts': ['Minor', 'Moderate', 'Major', 'Severe'],
            'assessment_statuses': ['Not Started', 'In Progress', 'Completed', 'Review Required']
        }
        return defaults.get(category, [])

    def _is_control_category(self, line: str) -> bool:
        """Check if a line represents a control category"""
        line = line.strip()
        # Match patterns like "5.1", "5.2", etc.
        return (
            len(line) > 2 and
            line[0].isdigit() and
            line[1] == '.' and
            line[2].isdigit() and
            ' ' in line  # Must have a description after the number
        )
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from text content"""
        items = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                item = line[1:].strip()
                if item:
                    items.append(item)
        return items

    def validate_date_format(self, date_str: str) -> bool:
        """Validate if date follows YYYY-MM-DD format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def validate_document_content(self, content_value: str) -> bool:
        """Validate Document Content field"""
        try:
            value = int(content_value)
            return value in [0, 1, 2]
        except ValueError:
            return False

    def validate_status(self, status: str) -> bool:
        """Validate Status field"""
        try:
            value = int(status)
            return value in [int(DocumentStatus.DRAFT), int(DocumentStatus.PUBLISHED)]
        except ValueError:
            return False

    def validate_portal_permissions(self, permissions: str) -> bool:
        """Validate Policy Portal Permissions field"""
        return permissions in [
            DocumentPermissions.PUBLIC,
            DocumentPermissions.PRIVATE,
            DocumentPermissions.CUSTOM_ROLES
        ]

    def validate_control_category(self, category: str) -> bool:
        """Validate if control category matches ISO 27001:2022 categories"""
        return any(c.lower() in category.lower() for c in self.iso_control_categories)

    def validate_risk_assessment(self, risk_data: Dict[str, str]) -> List[str]:
        """Validate risk assessment data according to ISO 27001:2022 requirements"""
        errors = []
        
        # Required risk fields
        risk_fields = {
            'Risk Level': self.valid_risk_levels,
            'Risk Likelihood': self.valid_risk_likelihoods,
            'Risk Impact': self.valid_risk_impacts
        }
        
        for field, valid_values in risk_fields.items():
            value = risk_data.get(field)
            if not value:
                errors.append(f"Missing required risk field: {field} (Suggestion: Ensure risk assessment follows ISO 27001:2022 risk management requirements)")
            elif value not in valid_values:
                errors.append(f"Invalid {field}: {value}. Must be one of: {', '.join(valid_values)}")
        
        # Validate Risk Assessment Status
        status = risk_data.get('Risk Assessment Status')
        if not status:
            errors.append("Missing required field: Risk Assessment Status (Mandatory for compliance) (Suggestion: Please provide a value for this mandatory field)")
        elif status not in self.valid_assessment_statuses:
            errors.append(f"Invalid Risk Assessment Status: {status}. Must be one of: {', '.join(self.valid_assessment_statuses)}")
        
        return errors

    def validate_policy(self, policy: Dict[str, str]) -> List[str]:
        """Validate a single policy against requirements"""
        errors = []
        
        # Validate control category
        control_category = policy.get('Control Category (Must match an ISO 27001:2022 control category)', '')
        if not control_category:
            errors.append("Missing required field: Control Category (Must match an ISO 27001:2022 control category) (Suggestion: Please provide a value for this mandatory field)")
        elif not self.validate_control_category(control_category):
            errors.append(f"Invalid Control Category: {control_category} (Suggestion: Please select a valid ISO 27001:2022 control category)")

        # Validate risk assessment fields
        risk_assessment_status = policy.get('Risk Assessment Status (Mandatory for compliance)', '')
        if not risk_assessment_status:
            errors.append("Missing required field: Risk Assessment Status (Mandatory for compliance) (Suggestion: Please provide a value for this mandatory field)")

        # Validate risk assessment components
        risk_fields = {
            'Risk Level': policy.get('Risk Level', ''),
            'Risk Likelihood': policy.get('Risk Likelihood', ''),
            'Risk Impact': policy.get('Risk Impact', '')
        }
        
        for field_name, value in risk_fields.items():
            if not value:
                errors.append(f"Risk Assessment: Missing required risk field: {field_name} (Suggestion: Ensure risk assessment follows ISO 27001:2022 risk management requirements)")

        # Validate review date
        review_date = policy.get('Next Review Date', '')
        pub_date = policy.get('Publication Date', '')
        if review_date and pub_date:
            try:
                review = datetime.strptime(review_date, '%Y-%m-%d')
                pub = datetime.strptime(pub_date, '%Y-%m-%d')
                if (review - pub).days > 365:
                    errors.append("Next Review Date: Review date is more than 1 year from publication date (Suggestion: ISO 27001:2022 recommends annual review cycles)")
            except ValueError:
                errors.append("Invalid date format. Use YYYY-MM-DD format.")

        return errors

    def _format_error_message(self, error: Dict[str, str]) -> str:
        """Format error message with field, error description, and suggestion"""
        return f"{error['field']}: {error['error']}\nSuggestion: {error['suggestion']}"

    def load_and_validate(self) -> None:
        """Load and validate all policies from the file input"""
        try:
            import io
            import codecs
            import docx
            from PyPDF2 import PdfReader

            # Handle different file types based on extension
            if isinstance(self.file_input, str):
                file_extension = self.file_input.lower().split('.')[-1]
                
                if file_extension == 'csv':
                    # Handle CSV files with NUL character handling
                    encodings = SUPPORTED_ENCODINGS
                    file_content = None
                    successful_encoding = None

                    for encoding in encodings:
                        try:
                            with open(self.file_input, 'r', encoding=encoding, errors='replace') as f:
                                file_content = f.read().replace('\0', '')  # Remove NUL characters
                                successful_encoding = encoding
                                break
                        except UnicodeDecodeError:
                            continue

                    if file_content is None:
                        raise ValueError("Unable to decode file with any supported encoding")

                    file = io.StringIO(file_content)
                    self._process_csv_file(file)

                elif file_extension in ['doc', 'docx']:
                    # Handle Word documents
                    doc = docx.Document(self.file_input)
                    policies = []
                    current_policy = {}
                    
                    for paragraph in doc.paragraphs:
                        text = paragraph.text.strip()
                        if not text:
                            continue
                            
                        # Check if this is a new policy section
                        if text.lower().endswith('policy') or any(text.lower().startswith(p.lower()) for p in ['scope', 'leadership', 'roles', 'risk', 'access', 'physical', 'change', 'backup', 'network', 'system', 'supplier', 'incident', 'business', 'compliance', 'security', 'training']):
                            if current_policy:
                                policies.append(current_policy)
                            current_policy = {
                                'Name (This field is mandatory.)': text,
                                'Control Category (Must match an ISO 27001:2022 control category)': self._determine_control_category(text),
                                'Risk Assessment Status (Mandatory for compliance)': 'In Progress',
                                'Risk Level': 'Medium',
                                'Risk Likelihood': 'Possible',
                                'Risk Impact': 'Moderate',
                                'Content Editor Text': ''
                            }
                        elif current_policy:
                            current_policy['Content Editor Text'] += text + '\n'
                    
                    # Add the last policy
                    if current_policy:
                        policies.append(current_policy)
                    
                    # Validate each policy
                    for policy in policies:
                        self.policies.append(policy)
                        errors = self.validate_policy(policy)
                        if errors:
                            self.errors.append({
                                'row': len(self.policies),
                                'policy_name': policy['Name (This field is mandatory.)'],
                                'errors': errors
                            })

                elif file_extension == 'pdf':
                    # Handle PDF files
                    reader = PdfReader(self.file_input)
                    text_content = ''
                    for page in reader.pages:
                        text_content += page.extract_text() + '\n'
                    self._process_text_content(text_content)

                elif file_extension == 'txt':
                    # Handle text files
                    with open(self.file_input, 'r', encoding='utf-8', errors='replace') as f:
                        text_content = f.read()
                    self._process_text_content(text_content)

        except Exception as e:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': [f'Error reading file: {str(e)}']})

    def _determine_control_category(self, policy_name: str) -> str:
        """Determine the ISO 27001 control category based on policy name"""
        policy_name = policy_name.lower()
        
        # Map common policy names to ISO 27001:2022 control categories
        category_mapping = {
            'information security policy': '5.1 Information security policies',
            'scope': '5.1 Information security policies',
            'leadership': '5.2 Information security roles and responsibilities',
            'roles': '5.2 Information security roles and responsibilities',
            'risk': '6.1 Risk assessment',
            'access': '8.1 Access control',
            'physical': '8.2 Physical security',
            'change': '8.3 Operations security',
            'backup': '8.3 Operations security',
            'network': '8.6 Technical vulnerability management',
            'system': '8.8 System development security',
            'supplier': '8.4 Third party security',
            'incident': '8.7 Information security incident management',
            'business continuity': '8.9 Business continuity',
            'compliance': '9.1 Compliance with requirements',
            'security review': '9.2 Information security reviews',
            'training': '7.3 Awareness and training'
        }
        
        # Find the best matching category
        for key, category in category_mapping.items():
            if key in policy_name:
                return category
        
        # Default to information security policies if no match found
        return '5.1 Information security policies'

    def _process_csv_file(self, file) -> None:
        """Process CSV file content"""
        try:
            reader = csv.DictReader(file)

            # Validate headers for structured CSV
            if not reader.fieldnames:
                raise ValueError("CSV file has no headers")

            # Clean and normalize fieldnames
            fieldnames = []
            for name in reader.fieldnames:
                # Remove BOM and whitespace
                cleaned = name.strip().replace('\ufeff', '').strip()
                fieldnames.append(cleaned)

            # Check for required headers using exact match
            missing_headers = []
            for required in REQUIRED_HEADERS:
                if required not in fieldnames:
                    missing_headers.append(required)

            if missing_headers:
                self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': [
                    f"CSV file is missing required headers: {', '.join(missing_headers)}. "
                    "Please provide a structured CSV with the required columns for validation."
                ]})
                return

            # Process rows
            for row_num, policy in enumerate(reader, start=2):
                cleaned_policy = {}
                for key, value in policy.items():
                    clean_key = key.strip().replace('\ufeff', '').strip()
                    clean_value = value.strip() if value else ''
                    cleaned_policy[clean_key] = clean_value
                self.policies.append(cleaned_policy)
                errors = self.validate_policy(cleaned_policy)
                if errors:
                    self.errors.append({
                        'row': row_num,
                        'policy_name': cleaned_policy.get('Name (This field is mandatory.)', 'Unknown'),
                        'errors': errors
                    })

        except csv.Error as e:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': [f'CSV parsing error: {str(e)}']})

    def _process_text_content(self, text_content: str) -> None:
        """Process text content from non-CSV files"""
        # Extract policy information from text content
        sections = self._extract_policy_sections(text_content)
        
        for section in sections:
            policy = {
                'Name (This field is mandatory.)': section.get('title', 'Untitled Policy'),
                'Control Category (Must match an ISO 27001:2022 control category)': section.get('category', '5.1 Information security policies'),
                'Content Editor Text': section.get('content', ''),
                'Risk Assessment Status (Mandatory for compliance)': 'Not Started',
                'Risk Level': 'Medium',
                'Risk Likelihood': 'Possible',
                'Risk Impact': 'Moderate',
                'Document Content': '0'
            }
            
            self.policies.append(policy)
            errors = self.validate_policy(policy)
            if errors:
                self.errors.append({
                    'row': len(self.policies),
                    'policy_name': policy['Name (This field is mandatory.)'],
                    'errors': errors
                })

    def _extract_policy_sections(self, text_content: str) -> List[Dict[str, str]]:
        """Extract policy sections from text content"""
        sections = []
        current_section = {'title': '', 'category': '', 'content': ''}
        lines = text_content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a potential section header
            if line.strip().lower().endswith('policy') or self._is_section_header(line):
                # Save previous section if it exists
                if current_section['content']:
                    sections.append(current_section.copy())
                current_section = {'title': line, 'category': '', 'content': ''}
                
                # Try to match with ISO control categories
                for category in self.iso_control_categories:
                    if any(keyword in line.lower() for keyword in category.lower().split()):
                        current_section['category'] = category
                        break
            else:
                current_section['content'] += line + '\n'
        
        # Add the last section
        if current_section['content']:
            sections.append(current_section)
            
        return sections

    def _is_section_header(self, line: str) -> bool:
        """Determine if a line is likely to be a section header"""
        # Common patterns for section headers
        header_patterns = [
            'policy',
            'procedure',
            'guideline',
            'standard',
            'requirements',
            'controls',
            'framework',
            'assessment',
            'compliance',
            'security'
        ]
        
        line = line.lower().strip()
        
        # Check if line matches common header patterns
        if any(pattern in line for pattern in header_patterns):
            # Additional checks to confirm it's a header
            # Headers are typically shorter than content
            if len(line.split()) <= 10:
                # Headers often start with capital letters
                if line[0].isupper():
                    return True
                    
        return False

    def print_validation_results(self) -> None:
        """Print validation results with enhanced formatting and suggestions"""
        print("\nISO 27001:2022 Security Policy Validation Results")
        print("=" * 50)
        
        if not self.errors:
            print("✅ All policies are valid according to ISO 27001:2022 requirements.")
            print(f"Total policies validated: {len(self.policies)}")
        else:
            print("❌ Found validation errors:")
            for error in self.errors:
                print(f"\nPolicy: {error['policy_name']} (Row {error['row']})")
                for err in error['errors']:
                    if isinstance(err, dict):
                        print(f"  • {self._format_error_message(err)}")
                    else:
                        print(f"  • {err}")
            
            print(f"\nTotal policies: {len(self.policies)}")
            print(f"Policies with errors: {len(self.errors)}")
            print("\nRecommendation: Review and update policies according to the suggestions above to ensure ISO 27001:2022 compliance.")

def main():
    validator = PolicyValidator('input files/Information-Security-Policytest_new.csv')
    validator.load_and_validate()
    validator.print_validation_results()

if __name__ == '__main__':
    main()