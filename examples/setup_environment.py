#!/usr/bin/env python3
"""
Environment Setup Script for API to Vector Examples

This script helps you set up the required environment variables and
verify your system is ready to run the API to vector examples.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def check_required_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'pandas',
        'asyncio',
        'asyncpg',  # For PostgreSQL
        'google-generativeai',  # For Gemini
        'aiohttp',  # For API calls
        'python-dotenv'  # For environment variables
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📥 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True


def check_postgresql():
    """Check PostgreSQL connection."""
    print("\n🐘 Checking PostgreSQL...")
    
    # Check if psql is available
    try:
        result = subprocess.run(['psql', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ PostgreSQL client found: {result.stdout.strip()}")
        else:
            print("❌ PostgreSQL client not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ PostgreSQL client not found in PATH")
        return False
    
    # Check environment variables
    db_vars = ['LOCAL_POSTGRES_HOST', 'LOCAL_POSTGRES_PORT', 'LOCAL_POSTGRES_DB', 'LOCAL_POSTGRES_USER', 'LOCAL_POSTGRES_PASSWORD']
    missing_vars = []
    
    for var in db_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing database environment variables: {missing_vars}")
        print("   Set them or use defaults (localhost:5432)")
    else:
        print("✅ Database environment variables configured")
    
    return True


def check_gemini_api():
    """Check Gemini API configuration."""
    print("\n🤖 Checking Gemini API...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return False
    
    if len(api_key) < 20:
        print("⚠️  GEMINI_API_KEY seems too short, please verify")
        return False
    
    print("✅ GEMINI_API_KEY configured")
    return True


def create_env_file():
    """Create a sample .env file."""
    print("\n📝 Creating sample .env file...")
    
    env_content = """# Database Configuration (used by dataload codebase)
LOCAL_POSTGRES_HOST=localhost
LOCAL_POSTGRES_PORT=5432
LOCAL_POSTGRES_DB=vector_db
LOCAL_POSTGRES_USER=postgres
LOCAL_POSTGRES_PASSWORD=your_password_here

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Logging Level
LOG_LEVEL=INFO
"""
    
    env_file = Path('.env')
    if env_file.exists():
        print("⚠️  .env file already exists, not overwriting")
        return
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file template")
    print("   Please edit .env file with your actual credentials")


def setup_postgresql_instructions():
    """Provide PostgreSQL setup instructions."""
    print("\n🐘 PostgreSQL Setup Instructions:")
    print("=" * 50)
    
    print("1. Install PostgreSQL:")
    print("   - Windows: Download from https://www.postgresql.org/download/windows/")
    print("   - macOS: brew install postgresql")
    print("   - Ubuntu: sudo apt-get install postgresql postgresql-contrib")
    
    print("\n2. Install pgvector extension:")
    print("   - Follow instructions at: https://github.com/pgvector/pgvector")
    print("   - Or use Docker: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password pgvector/pgvector:pg16")
    
    print("\n3. Create database:")
    print("   createdb vector_db")
    print("   psql vector_db -c 'CREATE EXTENSION vector;'")
    
    print("\n4. Test connection:")
    print("   psql -h localhost -p 5432 -U postgres -d vector_db")


def setup_gemini_instructions():
    """Provide Gemini API setup instructions."""
    print("\n🤖 Gemini API Setup Instructions:")
    print("=" * 50)
    
    print("1. Get API Key:")
    print("   - Visit: https://makersuite.google.com/app/apikey")
    print("   - Sign in with Google account")
    print("   - Create new API key")
    
    print("\n2. Set environment variable:")
    print("   - Linux/macOS: export GEMINI_API_KEY='your-api-key-here'")
    print("   - Windows: set GEMINI_API_KEY=your-api-key-here")
    print("   - Or add to .env file: GEMINI_API_KEY=your-api-key-here")
    
    print("\n3. Test API access:")
    print("   python -c \"import google.generativeai as genai; genai.configure(api_key='your-key'); print('API configured')\"")


def run_test_connection():
    """Test database and API connections."""
    print("\n🧪 Testing Connections...")
    
    # Test database connection
    try:
        import asyncpg
        import asyncio
        
        async def test_db():
            try:
                conn = await asyncpg.connect(
                    host=os.getenv('LOCAL_POSTGRES_HOST', 'localhost'),
                    port=int(os.getenv('LOCAL_POSTGRES_PORT', 5432)),
                    database=os.getenv('LOCAL_POSTGRES_DB', 'vector_db'),
                    user=os.getenv('LOCAL_POSTGRES_USER', 'postgres'),
                    password=os.getenv('LOCAL_POSTGRES_PASSWORD', 'password')
                )
                await conn.close()
                print("✅ Database connection successful")
                return True
            except Exception as e:
                print(f"❌ Database connection failed: {e}")
                return False
        
        db_ok = asyncio.run(test_db())
    except ImportError:
        print("⚠️  Cannot test database (asyncpg not installed)")
        db_ok = False
    
    # Test Gemini API
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            # Try to list models to test API access
            try:
                models = list(genai.list_models())
                print("✅ Gemini API connection successful")
                api_ok = True
            except Exception as e:
                print(f"❌ Gemini API connection failed: {e}")
                api_ok = False
        else:
            print("❌ Gemini API key not configured")
            api_ok = False
    except ImportError:
        print("⚠️  Cannot test Gemini API (google-generativeai not installed)")
        api_ok = False
    
    return db_ok and api_ok


def main():
    """Main setup function."""
    print("🚀 API to Vector Store Environment Setup")
    print("=" * 50)
    
    # Check system requirements
    python_ok = check_python_version()
    packages_ok = check_required_packages()
    postgres_ok = check_postgresql()
    gemini_ok = check_gemini_api()
    
    # Create .env file if needed
    create_env_file()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"   Python: {'✅' if python_ok else '❌'}")
    print(f"   Packages: {'✅' if packages_ok else '❌'}")
    print(f"   PostgreSQL: {'✅' if postgres_ok else '❌'}")
    print(f"   Gemini API: {'✅' if gemini_ok else '❌'}")
    
    if not all([python_ok, packages_ok, postgres_ok, gemini_ok]):
        print("\n⚠️  Some requirements are missing. See instructions below:")
        
        if not postgres_ok:
            setup_postgresql_instructions()
        
        if not gemini_ok:
            setup_gemini_instructions()
        
        print("\n📖 After setup, run this script again to verify configuration")
        return False
    
    # Test connections if everything looks good
    print("\n🧪 All requirements met! Testing connections...")
    connections_ok = run_test_connection()
    
    if connections_ok:
        print("\n🎉 Environment setup complete!")
        print("   You can now run the API to vector examples:")
        print("   python examples/comprehensive_api_to_vector_example.py")
    else:
        print("\n⚠️  Environment configured but connections failed")
        print("   Please check your database and API credentials")
    
    return connections_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)