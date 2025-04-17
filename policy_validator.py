import csv
import datetime
from typing import Dict, List, Optional

class PolicyValidator:
    def __init__(self, file_input):
        self.file_input = file_input
        self.policies = []
        self.errors = []
        self.iso_control_categories = [
            "5.1 Information security policies",
            "5.2 Information security roles and responsibilities",
            "5.3 Segregation of duties",
            "5.4 Management responsibilities",
            "5.5 Contact with authorities",
            "5.6 Contact with special interest groups",
            "5.7 Information security in project management",
            "5.8 Inventory of information and other associated assets",
            "5.9 Acceptable use of information and other associated assets",
            "5.10 Return of assets",
            "5.11 Classification of information",
            "5.12 Labelling of information",
            "5.13 Transfer of information",
            "5.14 Access control",
            "5.15 Identity management",
            "5.16 Authentication information",
            "5.17 Access rights",
            "5.18 Information security in supplier relationships",
            "5.19 Addressing information security within supplier agreements",
            "5.20 Managing information security in the ICT supply chain",
            "5.21 Monitoring, review and change management of supplier services",
            "5.22 Information security for use of cloud services",
            "5.23 Information security incident management planning and preparation",
            "5.24 Assessment and decision on information security events",
            "5.25 Response to information security incidents",
            "5.26 Learning from information security incidents",
            "5.27 Collection of evidence",
            "5.28 Information security during disruption",
            "5.29 ICT readiness for business continuity",
            "5.30 Legal, statutory, regulatory and contractual requirements",
            "5.31 Intellectual property rights",
            "5.32 Protection of records",
            "5.33 Privacy and protection of personal information",
            "5.34 Independent review of information security",
            "5.35 Compliance with policies and standards",
            "5.36 Documented operating procedures",
            "5.37 Information security during disruption"
        ]
        
        # Define valid risk assessment values
        self.valid_risk_levels = ['Low', 'Medium', 'High', 'Critical']
        self.valid_risk_likelihoods = ['Unlikely', 'Possible', 'Likely', 'Very Likely']
        self.valid_risk_impacts = ['Minor', 'Moderate', 'Major', 'Severe']
        self.valid_assessment_statuses = ['Not Started', 'In Progress', 'Completed', 'Review Required']

    def validate_date_format(self, date_str: str) -> bool:
        """Validate if date follows YYYY-MM-DD format"""
        try:
            datetime.datetime.strptime(date_str, '%Y-%m-%d')
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
            return value in [0, 1]
        except ValueError:
            return False

    def validate_portal_permissions(self, permissions: str) -> bool:
        """Validate Policy Portal Permissions field"""
        return permissions in ['public', 'private', 'custom-roles']

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
        elif control_category not in self.iso_control_categories:
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
                    errors.append("Next Review Date (Mandatory, it must follow the format YYYY-MM-DD, not the '-' character is used as delimiter.): Review date is more than 1 year from publication date (Suggestion: ISO 27001:2022 recommends annual review cycles)")
            except ValueError:
                errors.append("Invalid date format. Use YYYY-MM-DD format.")

        return errors

    def _format_error_message(self, error: Dict[str, str]) -> str:
        """Format error message with field, error description, and suggestion"""
        return f"{error['field']}: {error['error']}\nSuggestion: {error['suggestion']}"

    def load_and_validate(self) -> None:
        """Load and validate all policies from the CSV file"""
        try:
            # Handle different types of file inputs
            if isinstance(self.file_input, str):
                file = open(self.file_input, 'r', encoding='utf-8', newline='')
            else:
                import io
                content = self.file_input.getvalue().decode('utf-8')
                file = io.StringIO(content)

            try:
                reader = csv.DictReader(
                    file,
                    quoting=csv.QUOTE_ALL,
                    delimiter=',',
                    doublequote=True
                )
                # Validate that we have headers
                if not reader.fieldnames:
                    raise ValueError("CSV file has no headers")

                # Check for required headers
                required_headers = [
                    'Name (This field is mandatory.)',
                    'Control Category (Must match an ISO 27001:2022 control category)',
                    'Risk Assessment Status (Mandatory for compliance)',
                    'Risk Level',
                    'Risk Likelihood',
                    'Risk Impact'
                ]
                missing_headers = [h for h in required_headers if h not in reader.fieldnames]
                if missing_headers:
                    self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': [
                        f"CSV file is missing required headers: {', '.join(missing_headers)}. "
                        "Please provide a structured CSV with the required columns for validation. "
                        "If your file is a narrative policy document, this tool cannot validate it as a table of policies."
                    ]})
                    return

                for row_num, policy in enumerate(reader, start=2):
                    cleaned_policy = {}
                    for key, value in policy.items():
                        clean_key = key.strip().strip('\ufeff')
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
            finally:
                if isinstance(self.file_input, str):
                    file.close()
        except FileNotFoundError:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': ['CSV file not found']})
        except Exception as e:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': [f'Error reading CSV file: {str(e)}']})

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