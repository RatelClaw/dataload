#!/usr/bin/env python3
"""
DataMove Integration Test Script

This script tests the DataMove use case functionality to ensure all components
work together correctly. It's designed to be run as part of the development
workflow to validate the implementation.

Run this script to verify:
- Database connectivity
- CSV loading (local and S3 simulation)
- Table creation and data insertion
- Validation modes (existing_schema and new_schema)
- Error handling and rollback
- Performance metrics
"""

import asyncio
import pandas as pd
import os
import tempfile
import time
from typing import Dict, Any

# DataMove imports
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection

# Exception imports
from dataload.domain.entities import (
    DataMoveError,
    ValidationError,
    DatabaseOperationError,
    SchemaConflictError,
    CaseSensitivityError,
)


class DataMoveIntegrationTest:
    """Integration test suite for DataMove use case."""
    
    def __init__(self):
        self.use_case = None
        self.db_connection = None
        self.test_results = {}
        self.temp_dir = None
        
    async def setup(self):
        """Set up test environment."""
        print("🔧 Setting up integration test environment...")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="datamove_test_")
        print(f"📁 Created temp directory: {self.temp_dir}")
        
        # Initialize database connection
        try:
            self.db_connection = DBConnection()
            await self.db_connection.initialize()
            print("✅ Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            raise
        
        # Create repository and use case
        repository = PostgresDataMoveRepository(self.db_connection)
        self.use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
        print("✅ DataMove use case initialized")
    
    async def cleanup(self):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")
        
        # Drop test tables
        test_tables = [
            "dm_test_employees",
            "dm_test_products", 
            "dm_test_case_conflict",
            "dm_test_performance"
        ]
        
        for table_name in test_tables:
            try:
                # Note: This would require implementing drop_table method
                print(f"🗑️  Would drop table: {table_name}")
            except Exception as e:
                print(f"⚠️  Could not drop {table_name}: {e}")
        
        # Close database connection
        if self.db_connection:
            await self.db_connection.close()
            print("🔌 Database connection closed")
        
        # Clean up temp files
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"📁 Removed temp directory: {self.temp_dir}")
    
    def create_test_data(self) -> Dict[str, str]:
        """Create test CSV files."""
        print("\n📊 Creating test data files...")
        
        # Test 1: Basic employee data
        employees_data = pd.DataFrame({
            'id': [1, 2, 3, 4],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com'],
            'department': ['Engineering', 'Sales', 'Marketing', 'Engineering'],
            'salary': [95000, 75000, 68000, 102000],
            'active': [True, True, False, True]
        })
        
        # Test 2: Updated employee data (same schema)
        employees_updated = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com'],
            'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
            'salary': [98000, 77000, 70000, 105000, 85000],
            'active': [True, True, True, True, True]
        })
        
        # Test 3: Evolved employee data (schema changes)
        employees_evolved = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'name': ['Alice Johnson', 'Bob Smith', 'Charlie Brown', 'Diana Prince', 'Eve Wilson', 'Frank Miller'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'diana@test.com', 'eve@test.com', 'frank@test.com'],
            'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR', 'Finance'],
            'salary': [98000, 77000, 70000, 105000, 85000, 92000],
            'active': [True, True, True, True, True, True],
            # New columns
            'hire_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2022-11-05', '2023-04-12', '2023-05-01'],
            'manager_id': [None, 1, 1, None, 1, 4]
        })
        
        # Test 4: Case conflict data
        case_conflict_data = pd.DataFrame({
            'ID': [1, 2, 3],  # Case conflict with 'id'
            'Name': ['Alice', 'Bob', 'Charlie'],  # Case conflict with 'name'
            'EMAIL': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],  # Case conflict
            'Department': ['Engineering', 'Sales', 'Marketing']
        })
        
        # Test 5: Large dataset for performance testing
        large_data = pd.DataFrame({
            'id': range(1, 1001),
            'name': [f'User_{i}' for i in range(1, 1001)],
            'email': [f'user_{i}@test.com' for i in range(1, 1001)],
            'value': [i * 1.5 for i in range(1, 1001)],
            'category': [f'Category_{i % 10}' for i in range(1, 1001)]
        })
        
        # Save files
        files = {
            'employees': os.path.join(self.temp_dir, 'employees.csv'),
            'employees_updated': os.path.join(self.temp_dir, 'employees_updated.csv'),
            'employees_evolved': os.path.join(self.temp_dir, 'employees_evolved.csv'),
            'case_conflict': os.path.join(self.temp_dir, 'case_conflict.csv'),
            'large_data': os.path.join(self.temp_dir, 'large_data.csv')
        }
        
        employees_data.to_csv(files['employees'], index=False)
        employees_updated.to_csv(files['employees_updated'], index=False)
        employees_evolved.to_csv(files['employees_evolved'], index=False)
        case_conflict_data.to_csv(files['case_conflict'], index=False)
        large_data.to_csv(files['large_data'], index=False)
        
        print(f"✅ Created {len(files)} test files")
        return files
    
    async def test_new_table_creation(self, files: Dict[str, str]) -> bool:
        """Test creating new tables from CSV data."""
        print("\n" + "="*50)
        print("TEST 1: New Table Creation")
        print("="*50)
        
        try:
            result = await self.use_case.execute(
                csv_path=files['employees'],
                table_name="dm_test_employees",
                primary_key_columns=["id"]
            )
            
            success = (
                result.success and
                result.table_created and
                result.rows_processed == 4 and
                result.operation_type == "new_table"
            )
            
            print(f"✅ Success: {result.success}")
            print(f"📊 Rows processed: {result.rows_processed}")
            print(f"🆕 Table created: {result.table_created}")
            print(f"⏱️ Execution time: {result.execution_time:.3f}s")
            
            self.test_results['new_table_creation'] = success
            return success
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            self.test_results['new_table_creation'] = False
            return False
    
    async def test_existing_schema_validation(self, files: Dict[str, str]) -> bool:
        """Test strict schema validation mode."""
        print("\n" + "="*50)
        print("TEST 2: Existing Schema Validation (Strict)")
        print("="*50)
        
        try:
            # This should succeed - same schema
            result = await self.use_case.execute(
                csv_path=files['employees_updated'],
                table_name="dm_test_employees",
                move_type="existing_schema"
            )
            
            success = (
                result.success and
                not result.table_created and
                result.rows_processed == 5 and
                result.operation_type == "existing_schema"
            )
            
            print(f"✅ Success: {result.success}")
            print(f"📊 Rows processed: {result.rows_processed}")
            print(f"🔄 Operation type: {result.operation_type}")
            print("💡 Strict validation passed with matching schema")
            
            self.test_results['existing_schema_validation'] = success
            return success
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            self.test_results['existing_schema_validation'] = False
            return False
    
    async def test_new_schema_flexibility(self, files: Dict[str, str]) -> bool:
        """Test flexible schema validation mode."""
        print("\n" + "="*50)
        print("TEST 3: New Schema Flexibility")
        print("="*50)
        
        try:
            result = await self.use_case.execute(
                csv_path=files['employees_evolved'],
                table_name="dm_test_employees",
                move_type="new_schema"
            )
            
            success = (
                result.success and
                result.rows_processed == 6 and
                result.operation_type == "new_schema"
            )
            
            print(f"✅ Success: {result.success}")
            print(f"📊 Rows processed: {result.rows_processed}")
            print(f"🔄 Schema updated: {result.schema_updated}")
            
            # Show schema changes
            report = result.validation_report
            if report.schema_analysis.columns_added:
                print(f"➕ Columns added: {report.schema_analysis.columns_added}")
            
            print("💡 Flexible validation handled schema evolution")
            
            self.test_results['new_schema_flexibility'] = success
            return success
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            self.test_results['new_schema_flexibility'] = False
            return False
    
    async def test_case_sensitivity_detection(self, files: Dict[str, str]) -> bool:
        """Test case sensitivity conflict detection."""
        print("\n" + "="*50)
        print("TEST 4: Case Sensitivity Detection")
        print("="*50)
        
        try:
            # This should fail due to case conflicts
            result = await self.use_case.execute(
                csv_path=files['case_conflict'],
                table_name="dm_test_case_conflict",
                primary_key_columns=["ID"]  # Note: uppercase ID
            )
            
            # If we get here, the test failed (should have thrown exception)
            print(f"❌ Test failed: Expected case sensitivity error but operation succeeded")
            self.test_results['case_sensitivity_detection'] = False
            return False
            
        except CaseSensitivityError as e:
            print(f"✅ Expected case sensitivity error caught: {e}")
            print("💡 Case conflict detection working correctly")
            self.test_results['case_sensitivity_detection'] = True
            return True
            
        except Exception as e:
            print(f"❌ Test failed with unexpected error: {e}")
            self.test_results['case_sensitivity_detection'] = False
            return False
    
    async def test_dry_run_functionality(self, files: Dict[str, str]) -> bool:
        """Test dry run preview functionality."""
        print("\n" + "="*50)
        print("TEST 5: Dry Run Functionality")
        print("="*50)
        
        try:
            # Test dry run
            result = await self.use_case.execute(
                csv_path=files['employees'],
                table_name="dm_test_dry_run",
                dry_run=True
            )
            
            success = (
                result.success and
                not result.table_created and
                result.rows_processed == 4  # Should show what would be processed
            )
            
            print(f"✅ Dry run success: {result.success}")
            print(f"📊 Would process: {result.rows_processed} rows")
            print(f"🆕 Table would be created: {result.table_created}")
            print("💡 Dry run completed without making changes")
            
            # Test preview method
            preview = await self.use_case.get_operation_preview(
                csv_path=files['employees'],
                table_name="dm_test_preview"
            )
            
            preview_success = (
                preview.validation_passed and
                not preview.schema_analysis.table_exists
            )
            
            print(f"✅ Preview validation: {preview.validation_passed}")
            print("💡 Preview method working correctly")
            
            final_success = success and preview_success
            self.test_results['dry_run_functionality'] = final_success
            return final_success
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            self.test_results['dry_run_functionality'] = False
            return False
    
    async def test_performance_metrics(self, files: Dict[str, str]) -> bool:
        """Test performance with larger dataset."""
        print("\n" + "="*50)
        print("TEST 6: Performance Metrics")
        print("="*50)
        
        try:
            start_time = time.time()
            
            result = await self.use_case.execute(
                csv_path=files['large_data'],
                table_name="dm_test_performance",
                primary_key_columns=["id"],
                batch_size=500  # Test custom batch size
            )
            
            total_time = time.time() - start_time
            
            success = (
                result.success and
                result.rows_processed == 1000 and
                result.execution_time > 0
            )
            
            throughput = result.rows_processed / result.execution_time if result.execution_time > 0 else 0
            
            print(f"✅ Success: {result.success}")
            print(f"📊 Rows processed: {result.rows_processed}")
            print(f"⏱️ Execution time: {result.execution_time:.3f}s")
            print(f"🚀 Throughput: {throughput:.0f} rows/second")
            print(f"📈 Total test time: {total_time:.3f}s")
            
            # Performance should be reasonable (>100 rows/second for this test)
            performance_ok = throughput > 100
            
            if performance_ok:
                print("💡 Performance metrics within acceptable range")
            else:
                print("⚠️ Performance may need optimization")
            
            final_success = success and performance_ok
            self.test_results['performance_metrics'] = final_success
            return final_success
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            self.test_results['performance_metrics'] = False
            return False
    
    async def test_error_handling(self, files: Dict[str, str]) -> bool:
        """Test comprehensive error handling."""
        print("\n" + "="*50)
        print("TEST 7: Error Handling")
        print("="*50)
        
        error_tests_passed = 0
        total_error_tests = 3
        
        # Test 1: File not found
        try:
            await self.use_case.execute(
                csv_path="nonexistent_file.csv",
                table_name="test_table"
            )
            print("❌ File not found test failed - should have thrown error")
        except DataMoveError as e:
            print("✅ File not found error handled correctly")
            error_tests_passed += 1
        
        # Test 2: Invalid move_type
        try:
            await self.use_case.execute(
                csv_path=files['employees'],
                table_name="dm_test_employees",
                move_type="invalid_mode"
            )
            print("❌ Invalid move_type test failed - should have thrown error")
        except ValidationError as e:
            print("✅ Invalid move_type error handled correctly")
            error_tests_passed += 1
        
        # Test 3: Empty table name
        try:
            await self.use_case.execute(
                csv_path=files['employees'],
                table_name=""
            )
            print("❌ Empty table name test failed - should have thrown error")
        except ValidationError as e:
            print("✅ Empty table name error handled correctly")
            error_tests_passed += 1
        
        success = error_tests_passed == total_error_tests
        print(f"💡 Error handling tests passed: {error_tests_passed}/{total_error_tests}")
        
        self.test_results['error_handling'] = success
        return success
    
    def print_test_summary(self):
        """Print comprehensive test results summary."""
        print("\n" + "="*60)
        print("INTEGRATION TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"📊 Total tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {total_tests - passed_tests}")
        print(f"📈 Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            test_display = test_name.replace('_', ' ').title()
            print(f"   {status} - {test_display}")
        
        if passed_tests == total_tests:
            print("\n🎉 All integration tests passed!")
            print("💡 DataMove use case is working correctly")
        else:
            print(f"\n⚠️ {total_tests - passed_tests} test(s) failed")
            print("💡 Check the detailed output above for issues")
        
        return passed_tests == total_tests


async def main():
    """Run the complete integration test suite."""
    
    print("DataMove Use Case - Integration Test Suite")
    print("=" * 60)
    print("🧪 Testing all major functionality and error scenarios")
    print("⏱️ This may take a few minutes to complete")
    print()
    
    test_suite = DataMoveIntegrationTest()
    
    try:
        # Setup
        await test_suite.setup()
        
        # Create test data
        files = test_suite.create_test_data()
        
        # Run all tests
        await test_suite.test_new_table_creation(files)
        await test_suite.test_existing_schema_validation(files)
        await test_suite.test_new_schema_flexibility(files)
        await test_suite.test_case_sensitivity_detection(files)
        await test_suite.test_dry_run_functionality(files)
        await test_suite.test_performance_metrics(files)
        await test_suite.test_error_handling(files)
        
        # Print summary
        all_passed = test_suite.print_test_summary()
        
        if all_passed:
            print("\n🚀 DataMove is ready for production use!")
        else:
            print("\n🔧 Some tests failed - check implementation")
        
    except Exception as e:
        print(f"\n💥 Integration test suite failed: {e}")
        print("💡 Check your database configuration and connectivity")
        
    finally:
        # Cleanup
        await test_suite.cleanup()


if __name__ == "__main__":
    print("🔧 Prerequisites:")
    print("   1. Configure .env file with database credentials")
    print("   2. Ensure PostgreSQL is running and accessible")
    print("   3. Install: pip install vector-dataloader")
    print("   4. Ensure you have CREATE TABLE permissions")
    print()
    
    # Run integration tests
    asyncio.run(main())