#!/usr/bin/env python3
"""
FEATURE 5: DataMove - Schema Management & Validation

This demonstrates the DataMove use case for data migration WITHOUT embeddings.
Based on examples: data_move_example.py, datamove_simple_examples.py

Key Features:
1. New table creation (auto-detect schema)
2. Existing schema validation (strict mode)
3. New schema validation (flexible mode)
4. Dry-run validation
5. S3 and local file support

Prerequisites:
- Run 01_generate_test_data.py first
- PostgreSQL database
"""

import asyncio
import os
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.domain.entities import ValidationError, CaseSensitivityError


async def setup():
    """Initialize DataMove components."""
    print("🔧 Setting up DataMove...")
    
    db_conn = DBConnection()
    await db_conn.initialize()
    repo = PostgresDataMoveRepository(db_conn)
    
    # Create use case with auto-loader (from data_move_example.py)
    use_case = DataMoveUseCase.create_with_auto_loader(repository=repo)
    
    print("✅ DataMove initialized")
    return db_conn, use_case


# ==================== FEATURE 5.1: New Table Creation ====================

async def feature_5_1_new_table_creation(use_case):
    """
    Feature 5.1: New Table Creation (Auto-detect Schema)
    
    Based on: datamove_simple_examples.py
    - Automatically detects schema from CSV
    - No move_type parameter needed
    - Creates table with proper types
    """
    print("\n" + "="*70)
    print("FEATURE 5.1: New Table Creation (Auto-detect)")
    print("="*70)
    
    try:
        # Create new table (from datamove_simple_examples.py)
        result = await use_case.execute(
            csv_path='test_data/csv/employees_basic.csv',
            table_name='dm_employees_new',
            primary_key_columns=['id']  # Specify primary key
        )
        
        print(f"✅ Success!")
        print(f"   📊 Rows loaded: {result.rows_processed}")
        print(f"   🆕 Table created: dm_employees_new")
        print(f"   📋 Schema: Auto-detected from CSV")
        print(f"   🔑 Primary key: id")
        print(f"\n💡 Usage: No move_type needed for new tables")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ==================== FEATURE 5.2: Existing Schema (Strict) ====================

async def feature_5_2_existing_schema_strict(use_case):
    """
    Feature 5.2: Existing Schema Validation (Strict Mode)
    
    Based on: datamove_simple_examples.py
    - Requires EXACT schema match
    - Column names must match (case-sensitive)
    - Data types must match
    - Use for production with stable schemas
    """
    print("\n" + "="*70)
    print("FEATURE 5.2: Existing Schema (Strict Validation)")
    print("="*70)
    
    # Test 1: Matching schema (should succeed)
    print("\n🔹 Test 1: Matching Schema")
    try:
        result = await use_case.execute(
            csv_path='test_data/csv/employees_basic.csv',
            table_name='dm_employees_new',  # Existing table from 5.1
            move_type='existing_schema'  # Strict validation
        )
        
        print(f"   ✅ Success - exact schema match!")
        print(f"   📊 Rows updated: {result.rows_processed}")
        print(f"   🔒 Validation: Strict mode enforced")
        
    except ValidationError as e:
        print(f"   ❌ Validation failed: {str(e)[:100]}")
    
    # Test 2: Schema mismatch (should fail)
    print("\n🔹 Test 2: Schema Mismatch (Expected to Fail)")
    try:
        result = await use_case.execute(
            csv_path='test_data/csv/products.csv',  # Different schema
            table_name='dm_employees_new',
            move_type='existing_schema'
        )
        
        print(f"   ⚠️  Should have failed but didn't")
        
    except ValidationError as e:
        print(f"   ✅ Correctly rejected - schema mismatch detected!")
        print(f"   💡 existing_schema requires exact match")


# ==================== FEATURE 5.3: New Schema (Flexible) ====================

async def feature_5_3_new_schema_flexible(use_case):
    """
    Feature 5.3: New Schema Validation (Flexible Mode)
    
    Based on: datamove_simple_examples.py
    - Allows column additions/removals
    - Still prevents case-sensitivity conflicts
    - Use for development with evolving schemas
    """
    print("\n" + "="*70)
    print("FEATURE 5.3: New Schema (Flexible Validation)")
    print("="*70)
    
    # Create base table first
    print("🔹 Creating base table...")
    try:
        await use_case.execute(
            csv_path='test_data/csv/employees_basic.csv',
            table_name='dm_employees_flexible',
            primary_key_columns=['id']
        )
        print("   ✅ Base table created")
    except Exception as e:
        print(f"   ⚠️  Table may already exist: {e}")
    
    # Test schema evolution
    print("\n🔹 Test: Schema Evolution (Add/Remove Columns)")
    try:
        # Use different CSV with different columns
        result = await use_case.execute(
            csv_path='test_data/csv/products.csv',  # Different structure
            table_name='dm_employees_flexible',
            move_type='new_schema'  # Flexible validation
        )
        
        print(f"   ✅ Schema evolution handled!")
        print(f"   📊 Rows processed: {result.rows_processed}")
        print(f"   🔄 Schema updated: {result.schema_updated}")
        print(f"   💡 new_schema allows structural changes")
        
    except Exception as e:
        print(f"   ⚠️  Note: {str(e)[:150]}")
        print(f"   💡 This might fail due to incompatible schemas")


# ==================== FEATURE 5.4: Dry-Run Validation ====================

async def feature_5_4_dry_run(use_case):
    """
    Feature 5.4: Dry-Run Validation
    
    Based on: data_move_example.py
    - Preview operations without changes
    - Test validation before actual execution
    - Get recommendations
    """
    print("\n" + "="*70)
    print("FEATURE 5.4: Dry-Run Validation")
    print("="*70)
    
    # Dry run with existing_schema
    print("\n🔹 Dry Run: Existing Schema")
    try:
        result = await use_case.execute(
            csv_path='test_data/csv/employees_basic.csv',
            table_name='dm_employees_new',
            move_type='existing_schema',
            dry_run=True  # No actual changes
        )
        
        print(f"   ✅ Validation passed (no changes made)")
        print(f"   📊 Would process: {result.rows_processed} rows")
        print(f"   🧪 Dry run complete")
        
    except ValidationError as e:
        print(f"   ❌ Validation would fail: {str(e)[:100]}")
    
    # Preview using get_operation_preview
    print("\n🔹 Preview: Get Operation Details")
    try:
        preview = await use_case.get_operation_preview(
            csv_path='test_data/csv/products.csv',
            table_name='dm_employees_new',
            move_type='new_schema'
        )
        
        print(f"   ✅ Preview generated")
        print(f"   📋 Validation: {preview.validation_passed}")
        print(f"   📊 Schema Analysis:")
        print(f"      - Table exists: {preview.schema_analysis.table_exists}")
        print(f"      - Columns added: {len(preview.schema_analysis.columns_added)}")
        print(f"      - Columns removed: {len(preview.schema_analysis.columns_removed)}")
        
        if preview.recommendations:
            print(f"   💡 Recommendations:")
            for rec in preview.recommendations[:2]:
                print(f"      - {rec}")
        
    except Exception as e:
        print(f"   ⚠️  Preview: {str(e)[:100]}")


# ==================== FEATURE 5.5: Case Sensitivity ====================

async def feature_5_5_case_sensitivity(use_case):
    """
    Feature 5.5: Case Sensitivity Detection
    
    Prevents case conflicts like 'id' vs 'ID' that could cause issues.
    """
    print("\n" + "="*70)
    print("FEATURE 5.5: Case Sensitivity Detection")
    print("="*70)
    
    print("🔹 Test: Case Conflict Detection")
    try:
        # This would have case conflicts
        result = await use_case.execute(
            csv_path='test_data/csv/nested_users.csv',  # Different case
            table_name='dm_employees_new',
            move_type='new_schema'
        )
        
        print(f"   ⚠️  Should have detected case conflicts")
        
    except (ValidationError, CaseSensitivityError) as e:
        print(f"   ✅ Case conflict detected and prevented!")
        print(f"   💡 Protects against: ID vs id, Name vs name")


# ==================== FEATURE 5.6: S3 Integration ====================

async def feature_5_6_s3_integration(use_case):
    """
    Feature 5.6: S3 File Support
    
    Based on: data_move_example.py
    - Auto-detects S3 URIs (s3://)
    - Uses S3Loader automatically
    - Same interface as local files
    """
    print("\n" + "="*70)
    print("FEATURE 5.6: S3 Integration (Demo)")
    print("="*70)
    
    # Demo S3 URI pattern (from data_move_example.py)
    s3_uri = "s3://your-bucket/data/employees.csv"
    
    print(f"📁 S3 URI Pattern: {s3_uri}")
    print(f"💡 DataMove automatically detects S3 URIs")
    print(f"💡 create_with_auto_loader() handles this:")
    print(f"   - s3:// → S3Loader")
    print(f"   - local path → LocalLoader")
    print(f"\n📝 Example usage:")
    print(f"""
    result = await use_case.execute(
        csv_path='s3://bucket/file.csv',
        table_name='table',
        primary_key_columns=['id']
    )
    """)
    print(f"\n💡 Note: Requires AWS credentials configured")
    print(f"   - AWS CLI: aws configure")
    print(f"   - Environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")


# ==================== COMPARISON TABLE ====================

def print_comparison():
    """Print comparison of schema modes."""
    print("\n" + "="*70)
    print("📊 SCHEMA MODE COMPARISON")
    print("="*70)
    
    comparison = """
┌─────────────────────┬──────────────────┬──────────────────┐
│ Feature             │ existing_schema  │ new_schema       │
├─────────────────────┼──────────────────┼──────────────────┤
│ Exact match needed  │ ✅ Required      │ ❌ Flexible      │
│ Column additions    │ ❌ Not allowed   │ ✅ Allowed       │
│ Column removals     │ ❌ Not allowed   │ ✅ Allowed       │
│ Type changes        │ ❌ Not allowed   │ ⚠️  Validated    │
│ Case conflicts      │ ❌ Rejected      │ ❌ Rejected      │
│ Constraint checks   │ ✅ Strict        │ ✅ Strict        │
│ Use case            │ Production       │ Development      │
│ Data safety         │ Maximum          │ High             │
└─────────────────────┴──────────────────┴──────────────────┘
"""
    print(comparison)
    
    print("\n💡 When to use:")
    print("   🔒 existing_schema:")
    print("      - Production with stable schemas")
    print("      - Critical data migrations")
    print("      - Compliance requirements")
    
    print("\n   🔄 new_schema:")
    print("      - Development environments")
    print("      - Evolving data structures")
    print("      - ETL pipelines with changing sources")
    
    print("\n   🆕 No move_type (new table):")
    print("      - Initial data loads")
    print("      - Creating new tables")
    print("      - Auto schema detection")


# ==================== MAIN ====================

async def main():
    """Run all DataMove examples."""
    print("=" * 70)
    print("FEATURE 5: DataMove - Schema Management")
    print("=" * 70)
    print("\n📚 Based on library examples:")
    print("   - data_move_example.py")
    print("   - datamove_simple_examples.py")
    print("   - data_move_comprehensive_example.py")
    
    db_conn = None
    
    try:
        db_conn, use_case = await setup()
        
        # Run all features
        await feature_5_1_new_table_creation(use_case)
        await feature_5_2_existing_schema_strict(use_case)
        await feature_5_3_new_schema_flexible(use_case)
        await feature_5_4_dry_run(use_case)
        await feature_5_5_case_sensitivity(use_case)
        await feature_5_6_s3_integration(use_case)
        
        # Comparison
        print_comparison()
        
        print("\n" + "="*70)
        print("✅ Feature 5 Complete!")
        print("="*70)
        print("\n📊 Tables Created:")
        print("   - dm_employees_new (new table auto-detect)")
        print("   - dm_employees_flexible (flexible schema)")
        
        print("\n💡 Key Points:")
        print("   ✓ DataMove = Data migration WITHOUT embeddings")
        print("   ✓ existing_schema = Strict validation")
        print("   ✓ new_schema = Flexible validation")
        print("   ✓ dry_run = Preview before execution")
        print("   ✓ S3 support = Same interface as local files")
        
    except Exception as e:
        print(f"\n❌ Feature 5 failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if db_conn:
            await db_conn.close()
            print("\n🔌 Database connection closed")


if __name__ == "__main__":
    print("🚀 DataLoad Library - Feature 5 (DataMove)\n")
    asyncio.run(main())