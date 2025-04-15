import csv
import datetime
from typing import Dict, List, Optional

class PolicyValidator:
    def __init__(self, file_input):
        self.file_input = file_input
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
        
        # Clean up any quotes in field names for validation
        cleaned_policy = {}
        for key, value in policy.items():
            clean_key = key.strip().strip('"').strip()  # Remove quotes and whitespace
            cleaned_policy[clean_key] = value.strip() if value else ''

        # Required fields validation
        required_fields = [
            'Name (This field is mandatory.)',
            'Document Content (Mandatory, set one of the following values: 0 for Use Content, 1 for Use Attachments, 2 for Use URL)',
            'GRC Contact (Mandatory. Accepts multiple user logins or group names separated by "|". For User login use prefix "User-" and for Group name use "Group-". For example "User-admin|Group-Third Party Feedback|Group-Admin". You can get the login of an user account from System / Settings / User Management or name of a group from System / Settings / Groups.)',
            'Policy Reviewer Contact (Mandatory. Accepts multiple user logins or group names separated by "|". For User login use prefix "User-" and for Group name use "Group-". For example "User-admin|Group-Third Party Feedback|Group-Admin". You can get the login of an user account from System / Settings / User Management or name of a group from System / Settings / Groups.)',
            'Published Date (Mandatory, it must follow the format YYYY-MM-DD, not the "-" character is used as delimiter.)',
            'Next Review Date (Mandatory, it must follow the format YYYY-MM-DD, not the "-" character is used as delimiter.)',
            'Status (Mandatory, set value: 0 for Draft, 1 for Published)',
            'Document Type (You need to provide the name of the policy type, you can obtain the name of type from Control Catalogue / Policies / Settings / Document Type.)',
            'Version (This field is mandatory.)',
            'Policy Portal Permissions for this Document (Mandatory, set one of the following values: public for Public (Everyone can see the document), private for Private (Document is not shown on the portal), custom-roles for Only specific groups)'
        ]

        for field in required_fields:
            clean_field = field.strip().strip('"')  # Clean up field name for comparison
            if not cleaned_policy.get(clean_field) or cleaned_policy[clean_field].strip() == '':
                policy_errors.append(f"Missing required field: {field}")

        # Validate dates
        date_fields = [
            'Published Date (Mandatory, it must follow the format YYYY-MM-DD, not the "-" character is used as delimiter.)',
            'Next Review Date (Mandatory, it must follow the format YYYY-MM-DD, not the "-" character is used as delimiter.)'
        ]
        for date_field in date_fields:
            clean_field = date_field.strip().strip('"')
            if cleaned_policy.get(clean_field) and not self.validate_date_format(cleaned_policy[clean_field]):
                policy_errors.append(f"Invalid date format in {date_field}: {cleaned_policy[clean_field]}")

        # Validate Document Content
        doc_content_field = 'Document Content (Mandatory, set one of the following values: 0 for Use Content, 1 for Use Attachments, 2 for Use URL)'.strip().strip('"')
        doc_content = cleaned_policy.get(doc_content_field)
        if doc_content and not self.validate_document_content(doc_content):
            policy_errors.append("Invalid Document Content value. Must be 0, 1, or 2")

        # Validate Status
        status_field = 'Status (Mandatory, set value: 0 for Draft, 1 for Published)'.strip().strip('"')
        status = cleaned_policy.get(status_field)
        if status and not self.validate_status(status):
            policy_errors.append("Invalid Status value. Must be 0 or 1")

        # Validate Portal Permissions
        permissions_field = 'Policy Portal Permissions for this Document (Mandatory, set one of the following values: public for Public (Everyone can see the document), private for Private (Document is not shown on the portal), custom-roles for Only specific groups)'.strip().strip('"')
        permissions = cleaned_policy.get(permissions_field)
        if permissions and not self.validate_portal_permissions(permissions):
            policy_errors.append("Invalid Portal Permissions value. Must be 'public', 'private', or 'custom-roles'")

        # Validate custom roles when portal permissions is 'custom-roles'
        if permissions == 'custom-roles':
            custom_roles_field = 'Custom Roles (Mandatory in case you set permission to custom-roles, you can set more of following values: Owners for GRC Contact, Collaborators for Policy Reviewer Contact)'.strip().strip('"')
            custom_roles = cleaned_policy.get(custom_roles_field)
            if not custom_roles:
                policy_errors.append("Custom Roles is required when Portal Permissions is set to 'custom-roles'")

        return policy_errors

    def load_and_validate(self) -> None:
        """Load and validate all policies from the CSV file"""
        try:
            # Handle different types of file inputs
            if isinstance(self.file_input, str):
                file = open(self.file_input, 'r', encoding='utf-8', newline='')
            else:
                # For Streamlit's UploadedFile or other file-like objects
                # Convert to StringIO to ensure text mode
                import io
                content = self.file_input.getvalue().decode('utf-8')
                file = io.StringIO(content)

            try:
                # Configure CSV reader to properly handle quoted fields
                reader = csv.DictReader(
                    file,
                    quoting=csv.QUOTE_ALL,
                    delimiter=',',
                    doublequote=True
                )
                
                # Validate that we have headers
                if not reader.fieldnames:
                    raise ValueError("CSV file has no headers")
                    
                for row_num, policy in enumerate(reader, start=2):
                    # Clean policy data
                    cleaned_policy = {}
                    for key, value in policy.items():
                        # Remove any BOM characters and extra whitespace
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
                # Only close the file if it was opened by us (i.e., it was a file path)
                if isinstance(self.file_input, str):
                    file.close()
                    
        except FileNotFoundError:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': ['CSV file not found']})
        except Exception as e:
            self.errors.append({'row': 0, 'policy_name': 'N/A', 'errors': [f'Error reading CSV file: {str(e)}']})
            raise  # Re-raise to see full error details

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
    validator = PolicyValidator('input files/security-policies.csv')
    validator.load_and_validate()
    validator.print_validation_results()

if __name__ == '__main__':
    main()