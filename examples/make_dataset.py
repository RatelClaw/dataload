#!/usr/bin/env python3
"""
Test Data Generator for DataLoad Library Examples

Generates all test data needed for comprehensive library testing:
- CSV files for various scenarios
- Mock API JSON responses
- Sample data for transformations
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import random


def create_directory_structure():
    """Create organized directory structure for test data."""
    directories = [
        'test_data',
        'test_data/csv',
        'test_data/api_responses',
        'test_data/json'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


# ==================== CSV DATA GENERATORS ====================

def generate_basic_employee_csv():
    """Generate basic employee CSV for simple embedding scenarios."""
    data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
        'email': ['alice@company.com', 'bob@company.com', 'charlie@company.com', 
                  'diana@company.com', 'eve@company.com'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
        'bio': [
            'Senior software engineer with 5 years Python experience',
            'Sales manager specializing in enterprise solutions',
            'Marketing specialist with focus on digital campaigns',
            'Principal engineer leading ML initiatives',
            'HR manager handling recruitment and training'
        ],
        'skills': ['Python,SQL,Docker', 'Sales,CRM,Negotiation', 'Marketing,SEO,Analytics', 
                   'Python,ML,AWS', 'HR,Recruiting,Training'],
        'salary': [95000, 75000, 68000, 102000, 85000],
        'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12'],
        'active': [True, True, True, True, True]
    })
    
    path = 'test_data/csv/employees_basic.csv'
    data.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(data)} rows, {len(data.columns)} columns)")


def generate_products_csv():
    """Generate products CSV with rich descriptions for embeddings."""
    data = pd.DataFrame({
        'product_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'name': [
            'Wireless Headphones Pro',
            'Smart Fitness Watch',
            'Laptop Computer 15inch',
            'Portable Bluetooth Speaker',
            'Ergonomic Office Chair',
            'Standing Desk Electric',
            '4K Webcam HD',
            'Mechanical Keyboard RGB'
        ],
        'description': [
            'Premium noise-canceling wireless headphones with 30-hour battery life and superior sound quality',
            'Advanced fitness tracker with heart rate monitoring, GPS, and sleep tracking capabilities',
            'High-performance laptop with Intel i7 processor, 16GB RAM, and 512GB SSD for professional use',
            'Waterproof portable speaker with 360-degree sound, 20-hour battery, and wireless charging',
            'Ergonomic mesh office chair with lumbar support, adjustable armrests, and premium comfort',
            'Electric height-adjustable standing desk with memory presets and cable management system',
            '4K ultra HD webcam with auto-focus, noise-canceling microphone, and premium glass lens',
            'Mechanical gaming keyboard with RGB backlighting, hot-swappable switches, and aluminum frame'
        ],
        'category': ['Electronics', 'Wearables', 'Computers', 'Electronics', 
                     'Furniture', 'Furniture', 'Electronics', 'Electronics'],
        'price': [299.99, 249.99, 1299.99, 149.99, 449.99, 699.99, 179.99, 159.99],
        'in_stock': [True, True, True, False, True, True, True, True],
        'rating': [4.5, 4.3, 4.7, 4.6, 4.4, 4.8, 4.2, 4.5],
        'reviews_count': [1250, 890, 2340, 567, 423, 789, 234, 1123]
    })
    
    path = 'test_data/csv/products.csv'
    data.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(data)} rows, {len(data.columns)} columns)")


def generate_documents_csv():
    """Generate documents CSV for text embedding scenarios."""
    data = pd.DataFrame({
        'doc_id': list(range(1, 11)),
        'title': [
            'Introduction to Machine Learning',
            'Advanced Python Programming Techniques',
            'Database Design Best Practices',
            'Cloud Computing Architecture Patterns',
            'Microservices Design Principles',
            'RESTful API Development Guide',
            'DevOps and CI/CD Pipeline',
            'Data Science with Python',
            'Web Application Security',
            'Scalable System Design'
        ],
        'content': [
            'Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. This guide covers supervised learning, unsupervised learning, and neural networks.',
            'Advanced Python programming covers decorators, generators, context managers, metaclasses, and asynchronous programming patterns for building robust applications.',
            'Database design involves normalization, indexing strategies, query optimization, and choosing the right database type for your application needs.',
            'Cloud computing architecture patterns include microservices, serverless, container orchestration, and distributed system design principles.',
            'Microservices architecture breaks down applications into small, independent services that communicate via APIs, enabling scalability and maintainability.',
            'RESTful APIs follow architectural constraints including statelessness, client-server separation, cacheability, and uniform interface design.',
            'DevOps practices combine development and operations, focusing on automation, continuous integration, continuous deployment, and monitoring.',
            'Data science involves statistical analysis, data visualization, machine learning, and extracting insights from large datasets using Python.',
            'Web security covers authentication, authorization, encryption, SQL injection prevention, XSS protection, and CSRF mitigation.',
            'Scalable system design addresses load balancing, caching strategies, database sharding, message queues, and fault tolerance.'
        ],
        'category': [
            'AI/ML', 'Programming', 'Database', 'Cloud', 'Architecture',
            'API', 'DevOps', 'Data Science', 'Security', 'Architecture'
        ],
        'author': [
            'Dr. Sarah Chen', 'John Martinez', 'Emily Rodriguez', 'Michael Kim',
            'Jessica Taylor', 'David Brown', 'Lisa Anderson', 'Robert Lee',
            'Maria Garcia', 'James Wilson'
        ],
        'publish_date': [
            '2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05',
            '2024-05-12', '2024-06-08', '2024-07-22', '2024-08-15',
            '2024-09-03', '2024-10-11'
        ],
        'word_count': [850, 1200, 950, 1100, 1050, 900, 1150, 1300, 1000, 1250]
    })
    
    path = 'test_data/csv/documents.csv'
    data.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(data)} rows, {len(data.columns)} columns)")


def generate_nested_data_csv():
    """Generate CSV with nested column names for transformation testing."""
    data = pd.DataFrame({
        'user_id': [1, 2, 3, 4],
        'user_profile_first_name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'user_profile_last_name': ['Johnson', 'Smith', 'Brown', 'Prince'],
        'user_contact_email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com'],
        'user_contact_phone': ['+1-555-0101', '+1-555-0102', '+1-555-0103', '+1-555-0104'],
        'user_preferences_theme': ['dark', 'light', 'dark', 'light'],
        'user_preferences_language': ['en', 'es', 'en', 'fr']
    })
    
    path = 'test_data/csv/nested_users.csv'
    data.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(data)} rows, {len(data.columns)} columns)")


def generate_large_dataset_csv():
    """Generate large dataset for performance testing."""
    categories = ['Electronics', 'Furniture', 'Clothing', 'Books', 'Sports']
    
    data = pd.DataFrame({
        'id': range(1, 1001),
        'name': [f'Product {i:04d}' for i in range(1, 1001)],
        'description': [f'High-quality product with premium features and excellent customer reviews. Product number {i:04d}.' 
                       for i in range(1, 1001)],
        'category': [random.choice(categories) for _ in range(1000)],
        'price': [round(random.uniform(10, 1000), 2) for _ in range(1000)],
        'in_stock': [random.choice([True, False]) for _ in range(1000)]
    })
    
    path = 'test_data/csv/products_large.csv'
    data.to_csv(path, index=False)
    print(f"✅ Generated: {path} ({len(data)} rows, performance testing)")


# ==================== API RESPONSE GENERATORS ====================

def generate_device_api_response():
    """Generate mock device API response (similar to restful-api.dev)."""
    devices = [
        {
            "id": "1",
            "name": "Google Pixel 6 Pro",
            "data": {
                "color": "Cloudy White",
                "capacity": "128 GB",
                "price": 899.99,
                "year": 2021
            }
        },
        {
            "id": "2",
            "name": "Apple iPhone 12 Mini",
            "data": {
                "color": "Blue",
                "capacity": "256 GB",
                "price": 699.99,
                "year": 2020
            }
        },
        {
            "id": "3",
            "name": "Samsung Galaxy S21",
            "data": {
                "color": "Phantom Gray",
                "capacity": "128 GB",
                "price": 799.99,
                "year": 2021
            }
        },
        {
            "id": "4",
            "name": "Apple iPhone 13 Pro Max",
            "data": {
                "color": "Graphite",
                "capacity": "512 GB",
                "price": 1199.99,
                "year": 2021
            }
        },
        {
            "id": "5",
            "name": "Google Pixel 7",
            "data": {
                "color": "Obsidian",
                "capacity": "256 GB",
                "price": 599.99,
                "year": 2022
            }
        }
    ]
    
    path = 'test_data/api_responses/devices.json'
    with open(path, 'w') as f:
        json.dump(devices, f, indent=2)
    print(f"✅ Generated: {path} (mock API response)")


def generate_user_api_response():
    """Generate mock user API response with nested data."""
    users = [
        {
            "id": 1,
            "username": "alice_j",
            "profile": {
                "first_name": "Alice",
                "last_name": "Johnson",
                "age": 28,
                "bio": "Software engineer passionate about ML"
            },
            "contact": {
                "email": "alice@example.com",
                "phone": "+1-555-0101",
                "address": {
                    "city": "San Francisco",
                    "state": "CA",
                    "zip": "94105"
                }
            },
            "preferences": {
                "notifications": True,
                "newsletter": True,
                "theme": "dark"
            }
        },
        {
            "id": 2,
            "username": "bob_smith",
            "profile": {
                "first_name": "Bob",
                "last_name": "Smith",
                "age": 35,
                "bio": "Data scientist and AI researcher"
            },
            "contact": {
                "email": "bob@example.com",
                "phone": "+1-555-0102",
                "address": {
                    "city": "New York",
                    "state": "NY",
                    "zip": "10001"
                }
            },
            "preferences": {
                "notifications": False,
                "newsletter": True,
                "theme": "light"
            }
        }
    ]
    
    path = 'test_data/api_responses/users.json'
    with open(path, 'w') as f:
        json.dump(users, f, indent=2)
    print(f"✅ Generated: {path} (nested API response)")


def generate_posts_api_response():
    """Generate mock blog posts API response."""
    posts = [
        {
            "id": i,
            "title": f"Blog Post Title {i}",
            "content": f"This is the content of blog post {i}. It contains detailed information about various topics including technology, programming, and best practices. " * 3,
            "author": f"Author {i % 5}",
            "category": ["Technology", "Programming", "DevOps", "AI/ML", "Database"][i % 5],
            "published_date": (datetime(2024, 1, 1) + timedelta(days=i*7)).strftime('%Y-%m-%d'),
            "tags": [f"tag{j}" for j in range(i % 3 + 1)],
            "views": random.randint(100, 5000),
            "likes": random.randint(10, 500)
        }
        for i in range(1, 21)
    ]
    
    path = 'test_data/api_responses/posts.json'
    with open(path, 'w') as f:
        json.dump(posts, f, indent=2)
    print(f"✅ Generated: {path} (paginated API response)")


# ==================== JSON FILE GENERATORS ====================

def generate_complex_json():
    """Generate complex nested JSON for transformation testing."""
    data = {
        "company": {
            "name": "Tech Corp",
            "employees": [
                {
                    "id": 1,
                    "personal": {
                        "name": "Alice Johnson",
                        "contact": {
                            "email": "alice@techcorp.com",
                            "phones": ["+1-555-0101", "+1-555-0102"]
                        }
                    },
                    "employment": {
                        "position": "Senior Engineer",
                        "department": "Engineering",
                        "salary": 120000,
                        "start_date": "2020-01-15"
                    }
                },
                {
                    "id": 2,
                    "personal": {
                        "name": "Bob Smith",
                        "contact": {
                            "email": "bob@techcorp.com",
                            "phones": ["+1-555-0201"]
                        }
                    },
                    "employment": {
                        "position": "Product Manager",
                        "department": "Product",
                        "salary": 110000,
                        "start_date": "2021-03-20"
                    }
                }
            ]
        }
    }
    
    path = 'test_data/json/complex_nested.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Generated: {path} (complex nested JSON)")


def generate_array_json():
    """Generate JSON with arrays for transformation testing."""
    data = [
        {
            "id": 1,
            "name": "Product A",
            "features": ["feature1", "feature2", "feature3"],
            "ratings": [4.5, 4.7, 4.3, 4.8],
            "reviews": [
                {"user": "user1", "rating": 5},
                {"user": "user2", "rating": 4}
            ]
        },
        {
            "id": 2,
            "name": "Product B",
            "features": ["feature1", "feature4"],
            "ratings": [4.2, 4.1, 4.5],
            "reviews": [
                {"user": "user3", "rating": 4}
            ]
        }
    ]
    
    path = 'test_data/json/array_handling.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Generated: {path} (array handling JSON)")


def generate_device_unnested_api_response():
    """Generate mock device API response (UN-NESTED)."""
    devices = [
        {
            "id": "1",
            "name": "Google Pixel 6 Pro",
            "color": "Cloudy White",
            "capacity": "128 GB",
            "price": 899.99,
            "year": 2021
        },
        {
            "id": "2",
            "name": "Apple iPhone 12 Mini",
            "color": "Blue",
            "capacity": "256 GB",
            "price": 699.99,
            "year": 2020
        },
        {
            "id": "3",
            "name": "Samsung Galaxy S21",
            "color": "Phantom Gray",
            "capacity": "128 GB",
            "price": 799.99,
            "year": 2021
        },
        {
            "id": "4",
            "name": "Apple iPhone 13 Pro Max",
            "color": "Graphite",
            "capacity": "512 GB",
            "price": 1199.99,
            "year": 2021
        },
        {
            "id": "5",
            "name": "Google Pixel 7",
            "color": "Obsidian",
            "capacity": "256 GB",
            "price": 599.99,
            "year": 2022
        }
    ]
    
    path = 'test_data/api_responses/devices_unnested.json'
    with open(path, 'w') as f:
        json.dump(devices, f, indent=2)
    print(f"✅ Generated: {path} (unnested API response)")

# ==================== MAIN EXECUTION ====================

def main():
    """Generate all test data."""
    print("🚀 Generating Test Data for DataLoad Library")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    print()
    
    # Generate CSV files
    print("📊 Generating CSV Files...")
    generate_basic_employee_csv()
    generate_products_csv()
    generate_documents_csv()
    generate_nested_data_csv()
    generate_large_dataset_csv()
    print()
    
    # Generate API responses
    print("🌐 Generating Mock API Responses...")
    generate_device_api_response()
    generate_user_api_response()
    generate_posts_api_response()
    generate_device_unnested_api_response()
    print()
    
    # Generate JSON files
    print("📝 Generating JSON Files...")
    generate_complex_json()
    generate_array_json()
    print()
    
    print("=" * 60)
    print("✅ All test data generated successfully!")
    print("\n📁 Directory Structure:")
    print("test_data/")
    print("  ├── csv/               (CSV files for various scenarios)")
    print("  ├── api_responses/     (Mock API JSON responses)")
    print("  └── json/              (JSON files for transformations)")
    print("\n🎉 Ready to run examples!")


if __name__ == "__main__":
    main()