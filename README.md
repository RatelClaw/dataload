vector-dataloader
vector-dataloader is a robust and extensible Python library for loading CSV data from local files or AWS S3 into vector stores (Postgres, FAISS, Chroma) with embedding generation. It supports multiple embedding providers (AWS Bedrock, Google Gemini, Sentence-Transformers, OpenAI) and offers flexible embedding modes for scalable data processing. Visit the GitHub repository for source code and examples.
🚀 Features
Core Data Loading

Data Sources: Load data from local CSV files or AWS S3.
Embedding Modes: Supports combined or separated embedding generation.
Embedding Providers: AWS Bedrock, Google Gemini, Sentence-Transformers, OpenAI.
Vector Stores: Postgres (with pgvector), FAISS (in-memory), Chroma (persistent).
Data Updates: Handles new, updated, or removed rows with soft delete support.
Scalability: Batch processing, retries, and connection pooling for efficient operations.
Extensibility: Modular plugin-style architecture for providers and stores.
Validation: Enforces schema, type, and null checks for data integrity.

DataMove Use Case (New!)

Production-grade data migration from CSV to PostgreSQL without embedding generation.
Flexible schema validation with strict and flexible modes.
Automatic table creation and schema evolution.
Case-sensitivity conflict detection and prevention.
Comprehensive error handling with automatic rollback.
S3 integration with automatic loader selection.
Performance optimization with configurable batch processing.
Dry-run validation for previewing operations.

📦 Installation
Prerequisites

Python: Version 3.8 or higher.
Visual Studio Build Tools: Required for C++ dependencies (e.g., FAISS). Download from Visual Studio Build Tools and ensure the "Desktop development with C++" workload is installed.
pip or uv: Package manager for installing dependencies.

Step-by-Step Installation

Install the Core PackageInstall the minimal package without optional dependencies:
pip install vector-dataloader
# or
uv add vector-dataloader


Install Optional DependenciesInstall only the dependencies for the providers and stores you need:



Combination
Command
Notes



ChromaDB
pip install vector-dataloader[chroma]
Required for ChromaVectorStore.


FAISS
pip install vector-dataloader[faiss]
Required for FaissVectorStore.


Google Gemini
pip install vector-dataloader[gemini]
Required for GeminiEmbeddingProvider.


Sentence-Transformers
pip install vector-dataloader[sentence-transformers]
Required for SentenceTransformersProvider.


OpenAI
pip install vector-dataloader[openai]
Required for OpenAIProvider.


AWS Bedrock
pip install vector-dataloader[bedrock]
Required for BedrockEmbeddingProvider.


All Features
pip install vector-dataloader[all]
Installs all optional dependencies.


Example: To use Chroma with Gemini:
pip install vector-dataloader[chroma,gemini]


Verify InstallationConfirm the package is installed:
pip show vector-dataloader



⚙️ Usage
For detailed usage examples, refer to the examples folder in the GitHub repository: https://github.com/RatelClaw/dataload.  

Loading and Data Validation: See examples/test_all_scenarios.py for comprehensive scenarios covering data loading and validation.  
Embedding Features: Explore main_chroma_st.py, main_chroma_gemini.py, and other main files for specific embedding provider implementations.

All examples use asynchronous execution for efficiency. Ensure the input CSV file (e.g., data_to_load/sample.csv or data_to_load/sample_2.csv) exists with appropriate columns (e.g., id, name, description or Index, Name, Description).
🛠️ Configuration
Environment Variables
Configure the library using a .env file in your project root or system environment variables.Example .env:
# Google Gemini API Key
GOOGLE_API_KEY=your_google_api_key_here

# Postgres Configuration
LOCAL_POSTGRES_HOST=localhost
LOCAL_POSTGRES_PORT=5432
LOCAL_POSTGRES_DB=your_db_name
LOCAL_POSTGRES_USER=postgres
LOCAL_POSTGRES_PASSWORD=your_password

# AWS Configuration (for Bedrock/S3)
AWS_REGION=ap-southeast-1
SECRET_NAME=your_secret_name

Notes:

For AWS Bedrock or S3, configure AWS credentials using:aws configure
aws configure set aws_secret_access_key


Set use_aws=True in DBConnection for AWS integration.
Ensure the input CSV file matches the expected schema.

DataMove Use Case
The DataMove use case provides production-grade data migration from CSV files to PostgreSQL databases without embedding generation, ideal for ETL pipelines and data migration scenarios.
Quick Start
Refer to the examples/data_move_comprehensive_example.py for a production-ready example.
Key Features

Automatic Table Creation: Creates tables from CSV schema.
Schema Validation: Strict (existing_schema) and flexible (new_schema) modes.
S3 Integration: Automatic detection of S3 URIs.
Error Handling: Comprehensive validation with rollback on failures.
Performance: Configurable batch processing and memory management.
Dry Run: Preview operations without making changes.

Documentation

Quick Start Guide: Get started in 5 minutes.
API Documentation: Complete API reference.
Troubleshooting Guide: Common issues and solutions.
Comprehensive Examples: Production-ready examples.

📚 License
MIT LicenseCopyright (c) 2025 Shashwat Roy  
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.