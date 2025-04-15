import csv
import datetime
from typing import Dict, List, Optional

class PolicyValidator:
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.policies = []
        self.errors = []
        
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

    def validate_policy(self, policy: Dict[str, str]) -> List[str]:
        """Validate a single policy against ISO 27001:2022 requirements"""
        policy_errors = []
        
        # Required fields validation
        required_fields = [
            'Name (This field is mandatory.)',
            'Document Content (Mandatory, set one of the following values: 0 for Use Content, 1 for Use Attachments, 2 for Use URL)',
            'GRC Contact (Mandatory. Accepts multiple user logins or group names separated by ""|""...)',
            'Policy Reviewer Contact (Mandatory. Accepts multiple user logins or group names separated by ""|""...)',
            'Published Date (Mandatory, it must follow the format YYYY-MM-DD...)',
            'Next Review Date (Mandatory, it must follow the format YYYY-MM-DD...)',
            'Status (Mandatory, set value: 0 for Draft, 1 for Published)',
            'Document Type (You need to provide the name of the policy type...)',
            'Version (This field is mandatory.)',
            'Policy Portal Permissions for this Document (Mandatory...)'
        ]

        for field in required_fields:
            if not policy.get(field) or policy[field].strip() == '':
                policy_errors.append(f"Missing required field: {field}")

        # Validate dates
        for date_field in ['Published Date (Mandatory...)', 'Next Review Date (Mandatory...)']:
            if policy.get(date_field) and not self.validate_date_format(policy[date_field]):
                policy_errors.append(f"Invalid date format in {date_field}: {policy[date_field]}")

        # Validate Document Content
        doc_content = policy.get('Document Content (Mandatory...)')
        if doc_content and not self.validate_document_content(doc_content):
            policy_errors.append("Invalid Document Content value. Must be 0, 1, or 2")

        # Validate Status
        status = policy.get('Status (Mandatory...)')
        if status and not self.validate_status(status):
            policy_errors.append("Invalid Status value. Must be 0 or 1")

        # Validate Portal Permissions
        permissions = policy.get('Policy Portal Permissions for this Document (Mandatory...)')
        if permissions and not self.validate_portal_permissions(permissions):
            policy_errors.append("Invalid Portal Permissions value. Must be 'public', 'private', or 'custom-roles'")

        # Validate custom roles when portal permissions is 'custom-roles'
        if permissions == 'custom-roles':
            custom_roles = policy.get('Custom Roles (Mandatory in case...)')
            if not custom_roles:
                policy_errors.append("Custom Roles is required when Portal Permissions is set to 'custom-roles'")

        return policy_errors

    def load_and_validate(self) -> None:
        """Load and validate all policies from the CSV file"""
        try:
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row_num, policy in enumerate(reader, start=2):  # Start at 2 to account for header row
                    self.policies.append(policy)
                    errors = self.validate_policy(policy)
                    if errors:
                        self.errors.append({
                            'row': row_num,
                            'policy_name': policy.get('Name (This field is mandatory.)', 'Unknown'),
                            'errors': errors
                        })
        except FileNotFoundError:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': ['CSV file not found']})
        except Exception as e:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': [f'Error reading CSV file: {str(e)}']})

    def print_validation_results(self) -> None:
        """Print validation results"""
        print("\nISO 27001:2022 Security Policy Validation Results")
        print("=" * 50)
        
        if not self.errors:
            print("✅ All policies are valid according to ISO 27001:2022 requirements.")
            print(f"Total policies validated: {len(self.policies)}")
        else:
            print("❌ Found validation errors:")
            for error in self.errors:
                print(f"\nRow {error['row']} - Policy: {error['policy_name']}")
                for err in error['errors']:
                    print(f"  - {err}")
            print(f"\nTotal policies: {len(self.policies)}")
            print(f"Policies with errors: {len(self.errors)}")

def main():
    validator = PolicyValidator('security-policies.csv')
    validator.load_and_validate()
    validator.print_validation_results()

if __name__ == '__main__':
    main()