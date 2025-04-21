"""Configuration settings for the compliance analysis tool."""

from pathlib import Path
import os

# File paths
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
INPUT_FILES_DIR = BASE_DIR / "input files"

# Vector database settings
COLLECTION_NAME = "iso27001_2022"
CHROMA_SETTINGS = {
    "persist_directory": str(CHROMA_DB_DIR),
    "collection_name": COLLECTION_NAME,
}

# Document processing settings
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVERLAP = 200
MINIMUM_RELEVANCE_SCORE = 0.7
MAX_SEARCH_RESULTS = 10

# Analysis thresholds
COVERAGE_THRESHOLD = 0.8
MATURITY_IMPROVEMENT_THRESHOLD = 0.6

# File encodings to try
SUPPORTED_ENCODINGS = ['utf-8', 'utf-8-sig', 'cp1252', 'iso-8859-1']

# Document content types
class DocumentContent:
    USE_CONTENT = "0"
    USE_ATTACHMENTS = "1"
    USE_URL = "2"

# Document permissions
class DocumentPermissions:
    PUBLIC = "public"
    PRIVATE = "private"
    CUSTOM_ROLES = "custom-roles"

# Document statuses
class DocumentStatus:
    DRAFT = "0"
    PUBLISHED = "1"

# ISO 27001:2022 sections
ISO_SECTIONS = [
    "4 Context of the organization",
    "5 Leadership",
    "5.1 Information security policies",
    "5.2 Information security roles and responsibilities",
    "6 Planning",
    "6.1 Risk assessment",
    "7 Support",
    "7.3 Awareness and training",
    "8 Operation",
    "8.1 Access control",
    "8.2 Physical security",
    "8.3 Operations security",
    "8.4 Third party security",
    "8.6 Technical vulnerability management",
    "8.7 Information security incident management",
    "8.8 System development security",
    "8.9 Business continuity",
    "9 Performance evaluation",
    "9.1 Compliance with requirements",
    "9.2 Information security reviews",
    "10 Improvement"
]

# Required CSV headers
REQUIRED_HEADERS = [
    'Name (This field is mandatory.)',
    'Control Category (Must match an ISO 27001:2022 control category)',
    'Risk Assessment Status (Mandatory for compliance)',
    'Risk Level',
    'Risk Likelihood',
    'Risk Impact'
]

# Create required directories
for directory in [CACHE_DIR, CHROMA_DB_DIR, INPUT_FILES_DIR]:
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)