"""
Example usage of APIJSONStorageLoader.

This script demonstrates various ways to use the APIJSONStorageLoader
for loading and processing JSON data from different sources.
"""

import asyncio
import json
import tempfile
import os
from dataload.infrastructure.storage.api_json_loader import APIJSONStorageLoader


def example_basic_json_loading():
    """Example: Basic JSON data loading from dict."""
    print("=== Basic JSON Loading Example ===")
    
    # Sample JSON data
    sample_data = [
        {
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "profile": {
                "age": 30,
                "city": "New York",
                "preferences": {
                    "theme": "dark",
                    "notifications": True
                }
            },
            "tags": ["developer", "python", "api"]
        },
        {
            "id": 2,
            "name": "Jane Smith", 
            "email": "jane@example.com",
            "profile": {
                "age": 25,
                "city": "San Francisco",
                "preferences": {
                    "theme": "light",
                    "notifications": False
                }
            },
            "tags": ["designer", "ui/ux"]
        }
    ]
    
    # Create loader
    loader = APIJSONStorageLoader()
    
    # Load JSON data
    df = loader.load_json(sample_data)
    
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    print("Columns:", list(df.columns))
    print("\nFirst row:")
    print(df.iloc[0].to_dict())
    print()


def example_json_file_loading():
    """Example: Loading JSON from a file."""
    print("=== JSON File Loading Example ===")
    
    # Sample data to write to file
    file_data = {
        "users": [
            {"id": 1, "name": "Alice", "department": {"name": "Engineering", "floor": 3}},
            {"id": 2, "name": "Bob", "department": {"name": "Marketing", "floor": 2}}
        ]
    }
    
    # Create temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(file_data, f)
        json_path = f.name
    
    try:
        loader = APIJSONStorageLoader()
        
        # Load from file
        df = loader.load_json(json_path)
        
        print(f"Loaded from file: {len(df)} rows with {len(df.columns)} columns")
        print("Columns:", list(df.columns))
        print("\nData:")
        print(df.to_string())
        print()
        
    finally:
        os.unlink(json_path)


def example_column_mapping():
    """Example: Using column name mapping."""
    print("=== Column Mapping Example ===")
    
    data = [
        {
            "user_id": 1,
            "first_name": "John",
            "last_name": "Doe",
            "contact_info": {
                "email": "john@example.com",
                "phone": "555-1234"
            }
        }
    ]
    
    # Configuration with column mapping
    config = {
        'column_name_mapping': {
            'user_id': 'id',
            'first_name': 'fname',
            'last_name': 'lname',
            'contact_info_email': 'email',
            'contact_info_phone': 'phone'
        }
    }
    
    loader = APIJSONStorageLoader()
    df = loader.load_json(data, config)
    
    print("Original nested structure mapped to clean column names:")
    print("Columns:", list(df.columns))
    print("\nData:")
    print(df.to_string())
    print()


def example_data_transformations():
    """Example: Using data transformations."""
    print("=== Data Transformations Example ===")
    
    data = [
        {"first_name": "John", "last_name": "Doe", "salary": 75000, "start_year": 2020},
        {"first_name": "Jane", "last_name": "Smith", "salary": 85000, "start_year": 2019}
    ]
    
    # Configuration with transformations
    config = {
        'update_request_body_mapping': {
            'full_name': "concat({first_name}, ' ', {last_name})",
            'salary_k': "round({salary} / 1000)"
        },
        'fail_on_error': False  # Continue processing even if some transformations fail
    }
    
    loader = APIJSONStorageLoader()
    df = loader.load_json(data, config)
    
    print("Added computed fields:")
    print("Columns:", list(df.columns))
    print("\nData:")
    print(df.to_string())
    print()


def example_custom_flattening():
    """Example: Custom JSON flattening options."""
    print("=== Custom Flattening Example ===")
    
    data = [
        {
            "id": 1,
            "metadata": {
                "created": "2024-01-01",
                "tags": ["important", "urgent", "review"],
                "nested": {
                    "deep": {
                        "value": "deeply nested"
                    }
                }
            }
        }
    ]
    
    # Configuration with custom flattening
    config = {
        'separator': '__',  # Use double underscore
        'max_depth': 2,     # Limit nesting depth
        'handle_arrays': 'join'  # Join arrays instead of expanding
    }
    
    loader = APIJSONStorageLoader()
    df = loader.load_json(data, config)
    
    print("Custom flattening with __ separator and array joining:")
    print("Columns:", list(df.columns))
    print("\nData:")
    print(df.to_string())
    print()


def example_complete_workflow():
    """Example: Complete workflow with all features."""
    print("=== Complete Workflow Example ===")
    
    data = [
        {
            "employee_id": 1,
            "personal": {
                "first_name": "John",
                "last_name": "Doe",
                "contact": {
                    "email": "john@company.com",
                    "phone": "555-0001"
                }
            },
            "employment": {
                "position": "Senior Developer",
                "salary": 95000,
                "department": "Engineering",
                "skills": ["Python", "JavaScript", "SQL"]
            }
        }
    ]
    
    # Complete configuration
    config = {
        # Flattening options
        'separator': '_',
        'handle_arrays': 'join',
        
        # Data transformations
        'update_request_body_mapping': {
            'full_name': "concat({personal_first_name}, ' ', {personal_last_name})",
            'annual_salary_k': "round({employment_salary} / 1000)"
        },
        
        # Column mapping
        'column_name_mapping': {
            'employee_id': 'id',
            'personal_contact_email': 'email',
            'employment_position': 'job_title',
            'employment_department': 'dept',
            'employment_skills': 'skills',
            'full_name': 'name'
        }
    }
    
    loader = APIJSONStorageLoader()
    df = loader.load_json(data, config)
    
    print("Complete workflow: flatten → transform → map")
    print("Final columns:", list(df.columns))
    print("\nFinal data:")
    print(df.to_string())
    print()


def example_csv_compatibility():
    """Example: CSV loading for backward compatibility."""
    print("=== CSV Compatibility Example ===")
    
    # Create temporary CSV file
    csv_content = """id,name,email,age
1,John Doe,john@example.com,30
2,Jane Smith,jane@example.com,25
3,Bob Johnson,bob@example.com,35"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = f.name
    
    try:
        loader = APIJSONStorageLoader()
        
        # Load CSV using the same loader
        df = loader.load_csv(csv_path)
        
        print("CSV loading works with the same loader:")
        print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        print("Columns:", list(df.columns))
        print("\nData:")
        print(df.to_string())
        print()
        
    finally:
        os.unlink(csv_path)


def main():
    """Run all examples."""
    print("APIJSONStorageLoader Examples")
    print("=" * 50)
    print()
    
    example_basic_json_loading()
    example_json_file_loading()
    example_column_mapping()
    example_data_transformations()
    example_custom_flattening()
    example_complete_workflow()
    example_csv_compatibility()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    main()