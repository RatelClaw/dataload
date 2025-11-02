#!/usr/bin/env python3
"""
Complete DataMove Scenario Testing

This script tests all DataMove scenarios using generated test data.
Run generate_test_data.py first to create the required CSV files.

Scenarios covered:
1. New table creation
2. Existing schema validation (strict)
3. New schema validation (flexible)
4. Dry-run validation
5. Case sensitivity conflict detection
6. Data type mismatch handling
7. Constraint violation detection
8. Empty dataset handling
9. Large dataset performance
10. Error handling and rollback
"""

import asyncio
import time
from dataload.application.use_cases.data_move_use_case import DataMoveUseCase
from dataload.infrastructure.db.postgres_data_move_repository import PostgresDataMoveRepository
from dataload.infrastructure.db.db_connection import DBConnection
from dataload.domain.entities import (
    DataMoveError,
    ValidationError,
    DatabaseOperationError,
    SchemaConflictError,
    CaseSensitivityError,
)


class DataMoveTestRunner:
    """Test runner for all DataMove scenarios."""
    
    def __init__(self):
        self.use_case = None
        self.db_connection = None
        self.results = []
    
    async def setup(self):
        """Initialize DataMove components."""
        print("🔧 Setting up DataMove...")
        
        # Initialize database connection
        self.db_connection = DBConnection()
        await self.db_connection.initialize()
        
        # Create repository and use case
        repository = PostgresDataMoveRepository(self.db_connection)
        self.use_case = DataMoveUseCase.create_with_auto_loader(repository=repository)
        
        print("✅ DataMove initialized successfully")
    
    async def cleanup(self):
        """Clean up database connection."""
        if self.db_connection:
            await self.db_connection.close()
            print("🔌 Database connection closed")
    
    def log_result(self, scenario: str, success: bool, message: str, execution_time: float = 0):
        """Log test result."""
        status = "✅ PASS" if success else "❌ FAIL"
        self.results.append({
            'scenario': scenario,
            'success': success,
            'message': message,
            'execution_time': execution_time
        })
        print(f"{status} {scenario}: {message} ({execution_time:.3f}s)")
    
    async def test_1_new_table_creation(self):
        """Test 1: Create new table from CSV data."""
        print("\n" + "="*60)
        print("TEST 1: New Table Creation")
        print("="*60)
        
        start_time = time.time()
        try:
            # Create new table with base employee data
            result = await self.use_case.execute(
                csv_path="test_data/employees_base.csv",
                table_name="test_employees",  # New table name
                primary_key_columns=["id"]  # Specify primary key
            )
            
            execution_time = time.time() - start_time
            
            if result.success and result.table_created:
                self.log_result(
                    "New Table Creation",
                    True,
                    f"Created table with {result.rows_processed} rows",
                    execution_time
                )
            else:
                self.log_result(
                    "New Table Creation",
                    False,
                    "Table creation failed or not reported",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "New Table Creation",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_2_existing_schema_strict(self):
        """Test 2: Existing schema validation (strict mode)."""
        print("\n" + "="*60)
        print("TEST 2: Existing Schema Validation (Strict)")
        print("="*60)
        
        start_time = time.time()
        try:
            # Test with matching schema (should succeed)
            result = await self.use_case.execute(
                csv_path="test_data/employees_updated.csv",
                table_name="test_employees",  # Existing table from test 1
                move_type="existing_schema"  # Strict validation
            )
            
            execution_time = time.time() - start_time
            
            if result.success:
                self.log_result(
                    "Existing Schema (Matching)",
                    True,
                    f"Updated {result.rows_processed} rows",
                    execution_time
                )
            else:
                self.log_result(
                    "Existing Schema (Matching)",
                    False,
                    "Failed with matching schema",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Existing Schema (Matching)",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
        
        # Test with schema changes (should fail)
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/employees_evolved.csv",
                table_name="test_employees",
                move_type="existing_schema"
            )
            
            execution_time = time.time() - start_time
            # This should fail, so success=False is expected
            self.log_result(
                "Existing Schema (Schema Changes)",
                False,
                "Expected failure - schema changes not allowed",
                execution_time
            )
            
        except ValidationError as e:
            execution_time = time.time() - start_time
            # Expected validation error
            self.log_result(
                "Existing Schema (Schema Changes)",
                True,
                "Correctly rejected schema changes",
                execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Existing Schema (Schema Changes)",
                False,
                f"Unexpected exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_3_new_schema_flexible(self):
        """Test 3: New schema validation (flexible mode)."""
        print("\n" + "="*60)
        print("TEST 3: New Schema Validation (Flexible)")
        print("="*60)
        
        start_time = time.time()
        try:
            # Test with schema evolution (validation should pass, execution may fail due to unimplemented features)
            result = await self.use_case.execute(
                csv_path="test_data/employees_evolved.csv",
                table_name="test_employees",
                move_type="new_schema"  # Flexible validation
            )
            
            execution_time = time.time() - start_time
            
            if result.success:
                self.log_result(
                    "New Schema (Flexible)",
                    True,
                    f"Schema evolution successful: {result.rows_processed} rows",
                    execution_time
                )
            else:
                # Expected to fail due to unimplemented schema update
                self.log_result(
                    "New Schema (Flexible)",
                    False,
                    "Expected failure - schema update not implemented",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            # Expected due to unimplemented schema update
            self.log_result(
                "New Schema (Flexible)",
                False,
                f"Expected failure: {str(e)[:100]}",
                execution_time
            )
    
    async def test_4_dry_run_validation(self):
        """Test 4: Dry-run validation without making changes."""
        print("\n" + "="*60)
        print("TEST 4: Dry-Run Validation")
        print("="*60)
        
        # Test dry-run with existing_schema
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/employees_updated.csv",
                table_name="test_employees",
                move_type="existing_schema",
                dry_run=True  # No actual changes
            )
            
            execution_time = time.time() - start_time
            
            if result.success:
                self.log_result(
                    "Dry-Run (Existing Schema)",
                    True,
                    f"Validated {result.rows_processed} rows without changes",
                    execution_time
                )
            else:
                self.log_result(
                    "Dry-Run (Existing Schema)",
                    False,
                    "Dry-run validation failed",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Dry-Run (Existing Schema)",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
        
        # Test dry-run with new_schema
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/employees_evolved.csv",
                table_name="test_employees",
                move_type="new_schema",
                dry_run=True
            )
            
            execution_time = time.time() - start_time
            
            if result.success:
                self.log_result(
                    "Dry-Run (New Schema)",
                    True,
                    f"Schema evolution preview: {result.rows_processed} rows",
                    execution_time
                )
            else:
                self.log_result(
                    "Dry-Run (New Schema)",
                    False,
                    "Dry-run schema validation failed",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Dry-Run (New Schema)",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_5_case_sensitivity_conflicts(self):
        """Test 5: Case sensitivity conflict detection."""
        print("\n" + "="*60)
        print("TEST 5: Case Sensitivity Conflict Detection")
        print("="*60)
        
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/employees_case_conflict.csv",
                table_name="test_employees",
                move_type="new_schema"  # Should detect case conflicts
            )
            
            execution_time = time.time() - start_time
            # Should fail due to case conflicts
            self.log_result(
                "Case Sensitivity Conflicts",
                False,
                "Unexpected success - case conflicts should be detected",
                execution_time
            )
            
        except (ValidationError, CaseSensitivityError) as e:
            execution_time = time.time() - start_time
            # Expected validation error for case conflicts
            self.log_result(
                "Case Sensitivity Conflicts",
                True,
                "Correctly detected case conflicts",
                execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Case Sensitivity Conflicts",
                False,
                f"Unexpected exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_6_data_type_mismatches(self):
        """Test 6: Data type mismatch handling."""
        print("\n" + "="*60)
        print("TEST 6: Data Type Mismatch Handling")
        print("="*60)
        
        start_time = time.time()
        try:
            # Create new table for type mismatch testing
            result = await self.use_case.execute(
                csv_path="test_data/employees_type_mismatch.csv",
                table_name="test_employees_types",
                primary_key_columns=["id"]
            )
            
            execution_time = time.time() - start_time
            
            if result.success:
                self.log_result(
                    "Data Type Mismatches",
                    True,
                    f"Handled type mismatches: {result.rows_processed} rows",
                    execution_time
                )
            else:
                self.log_result(
                    "Data Type Mismatches",
                    False,
                    "Failed to handle type mismatches",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Data Type Mismatches",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_7_constraint_violations(self):
        """Test 7: Constraint violation detection."""
        print("\n" + "="*60)
        print("TEST 7: Constraint Violation Detection")
        print("="*60)
        
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/employees_constraint_violation.csv",
                table_name="test_employees_constraints",
                primary_key_columns=["id"]  # Will have null/duplicate violations
            )
            
            execution_time = time.time() - start_time
            # Should fail due to constraint violations
            self.log_result(
                "Constraint Violations",
                False,
                "Unexpected success - constraint violations should be detected",
                execution_time
            )
            
        except (ValidationError, DatabaseOperationError) as e:
            execution_time = time.time() - start_time
            # Expected validation/database error for constraint violations
            self.log_result(
                "Constraint Violations",
                True,
                "Correctly detected constraint violations",
                execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Constraint Violations",
                False,
                f"Unexpected exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_8_empty_dataset(self):
        """Test 8: Empty dataset handling."""
        print("\n" + "="*60)
        print("TEST 8: Empty Dataset Handling")
        print("="*60)
        
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/employees_empty.csv",
                table_name="test_employees_empty",
                primary_key_columns=["id"]
            )
            
            execution_time = time.time() - start_time
            
            if result.success and result.rows_processed == 0:
                self.log_result(
                    "Empty Dataset",
                    True,
                    "Successfully handled empty dataset",
                    execution_time
                )
            else:
                self.log_result(
                    "Empty Dataset",
                    False,
                    f"Unexpected result: {result.rows_processed} rows processed",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Empty Dataset",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_9_large_dataset_performance(self):
        """Test 9: Large dataset performance."""
        print("\n" + "="*60)
        print("TEST 9: Large Dataset Performance")
        print("="*60)
        
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/employees_large.csv",
                table_name="test_employees_large",
                primary_key_columns=["id"],
                batch_size=500  # Test batch processing
            )
            
            execution_time = time.time() - start_time
            
            if result.success and result.rows_processed == 1000:
                throughput = result.rows_processed / execution_time
                self.log_result(
                    "Large Dataset Performance",
                    True,
                    f"Processed 1000 rows at {throughput:.1f} rows/sec",
                    execution_time
                )
            else:
                self.log_result(
                    "Large Dataset Performance",
                    False,
                    f"Expected 1000 rows, got {result.rows_processed}",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Large Dataset Performance",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
    
    async def test_10_different_table_structure(self):
        """Test 10: Different table structure (products)."""
        print("\n" + "="*60)
        print("TEST 10: Different Table Structure")
        print("="*60)
        
        start_time = time.time()
        try:
            result = await self.use_case.execute(
                csv_path="test_data/products.csv",
                table_name="test_products",
                primary_key_columns=["product_id"]
            )
            
            execution_time = time.time() - start_time
            
            if result.success:
                self.log_result(
                    "Different Table Structure",
                    True,
                    f"Created products table: {result.rows_processed} rows",
                    execution_time
                )
            else:
                self.log_result(
                    "Different Table Structure",
                    False,
                    "Failed to create products table",
                    execution_time
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_result(
                "Different Table Structure",
                False,
                f"Exception: {str(e)[:100]}",
                execution_time
            )
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        total_time = sum(r['execution_time'] for r in self.results)
        print(f"⏱️  Total Execution Time: {total_time:.3f}s")
        
        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.results:
                if not result['success']:
                    print(f"   - {result['scenario']}: {result['message']}")
        
        print("\n💡 Note: Some failures are expected (e.g., constraint violations, case conflicts)")
        print("💡 These demonstrate the validation system working correctly")


async def main():
    """Run all DataMove scenario tests."""
    print("🚀 DataMove Complete Scenario Testing")
    print("="*60)
    print("Testing all DataMove functionality with generated test data")
    print("Run generate_test_data.py first if test_data/ directory doesn't exist")
    print()
    
    runner = DataMoveTestRunner()
    
    try:
        await runner.setup()
        
        # Run all tests
        await runner.test_1_new_table_creation()
        await runner.test_2_existing_schema_strict()
        await runner.test_3_new_schema_flexible()
        await runner.test_4_dry_run_validation()
        await runner.test_5_case_sensitivity_conflicts()
        await runner.test_6_data_type_mismatches()
        await runner.test_7_constraint_violations()
        await runner.test_8_empty_dataset()
        await runner.test_9_large_dataset_performance()
        await runner.test_10_different_table_structure()
        
        # Print summary
        runner.print_summary()
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())