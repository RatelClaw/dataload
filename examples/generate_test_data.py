#!/usr/bin/env python3
"""
Test Data Generator for DataMove Examples

This script generates various CSV files to test all DataMove scenarios:
- New table creation
- Existing schema validation (strict)
- New schema validation (flexible)
- Case sensitivity conflicts
- Data type mismatches
- Constraint violations
"""

import pandas as pd
import os
from datetime import datetime, timedelta


def create_test_data_directory():
    """Create test_data directory if it doesn't exist."""
    os.makedirs('test_data', exist_ok=True)
    print("📁 Created test_data/ directory")


def generate_base_employees_data():
    """Generate base employees data for new table creation."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 
                 'diana@company.com', 'eve@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
        'salary': [95000, 75000, 68000, 102000, 85000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12'],
        'active': [True, True, True, True, True],
        'skills': ['Python,SQL', 'Sales,CRM', 'Design,Adobe', 'Python,Leadership', 'HR,Recruiting']
    })
    
    data.to_csv('test_data/employees_base.csv', index=False)
    print("✅ Generated: employees_base.csv (8 columns, 5 rows)")
    return data


def generate_updated_employees_data():
    """Generate updated employees data with same schema (for existing_schema test)."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson', 'Frank Miller'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 
                 'diana@company.com', 'eve@company.com', 'frank@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR', 'Finance'],
        'salary': [98000, 77000, 70000, 105000, 87000, 92000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12', '2023-05-01'],
        'active': [True, True, True, True, True, True],
        'skills': ['Python,SQL,ML', 'Sales,CRM,Analytics', 'Design,Adobe,UX', 'Python,Leadership,AI', 'HR,Recruiting,Training', 'Finance,Excel,SAP']
    })
    
    data.to_csv('test_data/employees_updated.csv', index=False)
    print("✅ Generated: employees_updated.csv (8 columns, 6 rows)")
    return data


def generate_evolved_employees_data():
    """Generate evolved employees data with schema changes (for new_schema test)."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson', 'Frank Miller', 'Grace Lee'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 
                 'diana@company.com', 'eve@company.com', 'frank@company.com', 'grace@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR', 'Finance', 'Legal'],
        'salary': [98000, 77000, 70000, 105000, 87000, 92000, 110000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12', '2023-05-01', '2023-06-15'],
        'active': [True, True, True, True, True, True, True],
        # New columns added
        'manager_id': [None, 1, 1, 1, None, None, None],
        'phone': ['+1-555-0101', '+1-555-0102', '+1-555-0103', '+1-555-0104', '+1-555-0105', '+1-555-0106', '+1-555-0107'],
        'office_location': ['NYC', 'NYC', 'LA', 'NYC', 'Chicago', 'Chicago', 'SF']
        # Note: 'skills' column removed to test column removal
    })
    
    data.to_csv('test_data/employees_evolved.csv', index=False)
    print("✅ Generated: employees_evolved.csv (10 columns, 7 rows) - Added: manager_id, phone, office_location; Removed: skills")
    return data


def generate_case_conflict_data():
    """Generate data with case sensitivity conflicts."""
    data = pd.DataFrame({
        'ID': [1, 2, 3],  # Conflicts with 'id'
        'Name': ['Alice', 'Bob', 'Charlie'],  # Conflicts with 'name'
        'EMAIL': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],  # Conflicts with 'email'
        'Department': ['Engineering', 'Sales', 'Marketing'],  # Conflicts with 'department'
        'Salary': [95000, 75000, 68000]  # Conflicts with 'salary'
    })
    
    data.to_csv('test_data/employees_case_conflict.csv', index=False)
    print("✅ Generated: employees_case_conflict.csv (5 columns, 3 rows) - Case conflicts: ID, Name, EMAIL")
    return data


def generate_type_mismatch_data():
    """Generate data with data type mismatches."""
    data = pd.DataFrame({
        'id': ['A1', 'B2', 'C3'],  # String instead of integer
        'name': [123, 456, 789],  # Integer instead of string
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing'],
        'salary': ['high', 'medium', 'low'],  # String instead of integer
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10'],
        'active': ['yes', 'no', 'maybe'],  # String instead of boolean
        'skills': ['Python,SQL', 'Sales,CRM', 'Design,Adobe']
    })
    
    data.to_csv('test_data/employees_type_mismatch.csv', index=False)
    print("✅ Generated: employees_type_mismatch.csv (8 columns, 3 rows) - Type mismatches: id, name, salary, active")
    return data


def generate_constraint_violation_data():
    """Generate data with constraint violations (nulls, duplicates)."""
    data = pd.DataFrame({
        'id': [1, 1, None, 4, 5],  # Duplicate and null primary key
        'name': ['Alice Johnson', 'Bob Smith', None, 'Diana Prince', 'Eve Wilson'],  # Null name
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 
                 'diana@company.com', 'alice@company.com'],  # Duplicate email
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
        'salary': [95000, 75000, 68000, 102000, 85000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12'],
        'active': [True, True, True, True, True],
        'skills': ['Python,SQL', 'Sales,CRM', 'Design,Adobe', 'Python,Leadership', 'HR,Recruiting']
    })
    
    data.to_csv('test_data/employees_constraint_violation.csv', index=False)
    print("✅ Generated: employees_constraint_violation.csv (8 columns, 5 rows) - Violations: duplicate/null id, null name")
    return data


def generate_empty_data():
    """Generate empty CSV with headers only."""
    data = pd.DataFrame(columns=['id', 'name', 'email', 'department', 'salary', 'hire_date', 'active', 'skills'])
    data.to_csv('test_data/employees_empty.csv', index=False)
    print("✅ Generated: employees_empty.csv (8 columns, 0 rows) - Empty dataset")
    return data


def generate_large_dataset():
    """Generate larger dataset for performance testing."""
    import random
    
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Legal', 'Operations']
    skills_list = ['Python', 'SQL', 'JavaScript', 'React', 'Sales', 'CRM', 'Design', 'Adobe', 'Leadership', 'Analytics']
    
    data = []
    for i in range(1, 1001):  # 1000 rows
        data.append({
            'id': i,
            'name': f'Employee {i:04d}',
            'email': f'employee{i:04d}@company.com',
            'department': random.choice(departments),
            'salary': random.randint(50000, 150000),
            'hire_date': (datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460))).strftime('%Y-%m-%d'),
            'active': random.choice([True, False]),
            'skills': ','.join(random.sample(skills_list, random.randint(1, 3)))
        })
    
    df = pd.DataFrame(data)
    df.to_csv('test_data/employees_large.csv', index=False)
    print("✅ Generated: employees_large.csv (8 columns, 1000 rows) - Performance testing dataset")
    return df


def generate_products_data():
    """Generate products data for different table testing."""
    data = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Laptop Pro', 'Desktop Elite', 'Tablet Max', 'Phone Ultra', 'Watch Smart'],
        'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Wearables'],
        'price': [1299.99, 899.99, 599.99, 799.99, 299.99],
        'in_stock': [True, True, False, True, True],
        'description': ['High-performance laptop', 'Powerful desktop', 'Versatile tablet', 'Latest smartphone', 'Smart fitness watch']
    })
    
    data.to_csv('test_data/products.csv', index=False)
    print("✅ Generated: products.csv (6 columns, 5 rows) - Different table for testing")
    return data


def main():
    """Generate all test data files."""
    print("🚀 Generating test data for DataMove examples...")
    print("=" * 50)
    
    create_test_data_directory()
    
    # Generate all test datasets
    generate_base_employees_data()
    generate_updated_employees_data()
    generate_evolved_employees_data()
    generate_case_conflict_data()
    generate_type_mismatch_data()
    generate_constraint_violation_data()
    generate_empty_data()
    generate_large_dataset()
    generate_products_data()
    
    print("=" * 50)
    print("✅ All test data generated successfully!")
    print("\n📋 Generated files:")
    print("   - employees_base.csv: Base dataset for new table creation")
    print("   - employees_updated.csv: Updated data with same schema")
    print("   - employees_evolved.csv: Schema evolution (add/remove columns)")
    print("   - employees_case_conflict.csv: Case sensitivity conflicts")
    print("   - employees_type_mismatch.csv: Data type mismatches")
    print("   - employees_constraint_violation.csv: Constraint violations")
    print("   - employees_empty.csv: Empty dataset")
    print("   - employees_large.csv: Large dataset (1000 rows)")
    print("   - products.csv: Different table structure")
    
    print("\n💡 Use these files with test_all_scenarios.py to test DataMove functionality")


if __name__ == "__main__":
    main()