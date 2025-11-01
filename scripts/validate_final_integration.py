#!/usr/bin/env python3
"""
Final Integration Validation Script for APIJSONStorageLoader.

This script runs comprehensive validation tests to ensure the APIJSONStorageLoader
integrates properly with all existing components and maintains backward compatibility.

Usage:
    python scripts/validate_final_integration.py [--verbose] [--report-file output.json]
"""

import asyncio
import json
import os
import sys
import time
import traceback
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules and components
from tests.integration.test_final_integration_compatibility import *
from tests.integration.test_comprehensive_integration_validation import *


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration: float
    results: List[ValidationResult]
    summary: Dict[str, Any]


class IntegrationValidator:
    """Comprehensive integration validator."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
    
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    async def run_test(self, test_func, test_name: str, *args, **kwargs) -> ValidationResult:
        """Run a single test and capture results."""
        self.log(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func(*args, **kwargs)
            else:
                test_func(*args, **kwargs)
            
            duration = time.time() - start_time
            result = ValidationResult(
                test_name=test_name,
                success=True,
                duration=duration
            )
            self.log(f"✅ {test_name} passed ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            result = ValidationResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error_message=error_msg,
                details={"traceback": traceback.format_exc()}
            )
            self.log(f"❌ {test_name} failed ({duration:.2f}s): {error_msg}", "ERROR")
        
        self.results.append(result)
        return result
    
    async def validate_storage_loader_compatibility(self):
        """Validate storage loader compatibility."""
        self.log("=== Storage Loader Compatibility Tests ===")
        
        test_class = TestStorageLoaderCompatibility()
        
        # Run interface compliance test
        await self.run_test(
            test_class.test_storage_loader_interface_compliance,
            "storage_loader_interface_compliance"
        )
        
        # Create test data directly (not using fixtures)
        sample_csv_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com'],
            'department': ['Engineering', 'Sales', 'Marketing']
        })
        
        sample_json_data = [
            {
                'id': 1,
                'name': 'Alice',
                'email': 'alice@test.com',
                'department': 'Engineering',
                'profile': {'age': 30, 'city': 'New York'}
            },
            {
                'id': 2,
                'name': 'Bob',
                'email': 'bob@test.com',
                'department': 'Sales',
                'profile': {'age': 25, 'city': 'San Francisco'}
            }
        ]
        
        # Create temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_csv = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            temp_json = f.name
        
        try:
            # Run CSV compatibility test
            await self.run_test(
                test_class.test_csv_loading_compatibility,
                "csv_loading_compatibility",
                temp_csv, sample_csv_data
            )
            
            # Run JSON compatibility tests
            await self.run_test(
                test_class.test_json_loading_compatibility,
                "json_loading_compatibility",
                sample_json_data
            )
            
            await self.run_test(
                test_class.test_json_file_loading_compatibility,
                "json_file_loading_compatibility",
                temp_json, sample_json_data
            )
            
            # Run error handling test
            await self.run_test(
                test_class.test_error_handling_compatibility,
                "error_handling_compatibility"
            )
            
            # Run S3 URI handling test
            await self.run_test(
                test_class.test_s3_uri_handling,
                "s3_uri_handling"
            )
            
        finally:
            # Clean up temporary files
            os.unlink(temp_csv)
            os.unlink(temp_json)
    
    async def validate_use_case_compatibility(self):
        """Validate use case compatibility."""
        self.log("=== Use Case Compatibility Tests ===")
        
        test_class = TestUseCaseCompatibility()
        
        # Create test data directly
        mock_repository = MockRepository()
        mock_embedding_service = MockEmbeddingProvider()
        sample_csv_data = pd.DataFrame({
            'id': [1, 2],
            'name': ['Alice', 'Bob'],
            'description': ['Engineer', 'Designer']
        })
        
        # Create temporary CSV file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_csv = f.name
        
        try:
            # Run dataload use case tests
            await self.run_test(
                test_class.test_dataload_use_case_with_api_loader,
                "dataload_use_case_with_api_loader",
                mock_repository, mock_embedding_service, temp_csv
            )
            
            await self.run_test(
                test_class.test_dataload_use_case_with_local_loader,
                "dataload_use_case_with_local_loader",
                mock_repository, mock_embedding_service, temp_csv
            )
            
            # Run DataAPIJSONUseCase test
            await self.run_test(
                test_class.test_data_api_json_use_case_integration,
                "data_api_json_use_case_integration",
                mock_repository, mock_embedding_service
            )
            
            # Run DataMoveUseCase test
            await self.run_test(
                test_class.test_data_move_use_case_auto_loader_compatibility,
                "data_move_use_case_auto_loader_compatibility",
                mock_repository
            )
            
        finally:
            os.unlink(temp_csv)
    
    async def validate_embedding_provider_compatibility(self):
        """Validate embedding provider compatibility."""
        self.log("=== Embedding Provider Compatibility Tests ===")
        
        test_class = TestEmbeddingProviderCompatibility()
        
        # Create test data directly
        mock_repository = MockRepository()
        mock_gemini_provider = MockEmbeddingProvider("gemini", 768)
        mock_openai_provider = MockEmbeddingProvider("openai", 1536)
        
        # Run embedding provider tests
        await self.run_test(
            test_class.test_gemini_provider_integration,
            "gemini_provider_integration",
            mock_repository, mock_gemini_provider
        )
        
        await self.run_test(
            test_class.test_openai_provider_integration,
            "openai_provider_integration",
            mock_repository, mock_openai_provider
        )
        
        await self.run_test(
            test_class.test_embedding_column_mapping_compatibility,
            "embedding_column_mapping_compatibility",
            mock_repository, mock_gemini_provider
        )
    
    async def validate_database_repository_compatibility(self):
        """Validate database repository compatibility."""
        self.log("=== Database Repository Compatibility Tests ===")
        
        test_class = TestDatabaseRepositoryCompatibility()
        
        # Create test data directly
        mock_postgres_repository = MockRepository()
        mock_embedding_service = MockEmbeddingProvider()
        
        # Run repository tests
        await self.run_test(
            test_class.test_postgres_repository_integration,
            "postgres_repository_integration",
            mock_postgres_repository, mock_embedding_service
        )
        
        await self.run_test(
            test_class.test_existing_table_upsert_compatibility,
            "existing_table_upsert_compatibility",
            mock_postgres_repository, mock_embedding_service
        )
        
        await self.run_test(
            test_class.test_vector_column_compatibility,
            "vector_column_compatibility",
            mock_postgres_repository, mock_embedding_service
        )
    
    async def validate_regression_tests(self):
        """Validate regression tests."""
        self.log("=== Regression Tests ===")
        
        test_class = TestRegressionTests()
        
        # Create test data directly
        sample_data_scenarios = {
            'simple_flat': [
                {'id': 1, 'name': 'Alice', 'age': 30},
                {'id': 2, 'name': 'Bob', 'age': 25}
            ],
            'nested_objects': [
                {
                    'id': 1,
                    'user': {'name': 'Alice', 'profile': {'age': 30, 'city': 'NYC'}},
                    'metadata': {'created': '2024-01-01', 'active': True}
                }
            ],
            'arrays': [
                {
                    'id': 1,
                    'name': 'Alice',
                    'skills': ['Python', 'JavaScript', 'SQL'],
                    'projects': [
                        {'name': 'Project A', 'status': 'completed'},
                        {'name': 'Project B', 'status': 'in_progress'}
                    ]
                }
            ],
            'mixed_types': [
                {
                    'id': 1,
                    'name': 'Alice',
                    'age': 30,
                    'salary': 75000.50,
                    'active': True,
                    'start_date': '2023-01-15',
                    'tags': None,
                    'metadata': {}
                }
            ]
        }
        
        # Run regression tests
        await self.run_test(
            test_class.test_json_flattening_regression,
            "json_flattening_regression",
            sample_data_scenarios
        )
        
        await self.run_test(
            test_class.test_column_mapping_regression,
            "column_mapping_regression",
            sample_data_scenarios
        )
        
        await self.run_test(
            test_class.test_data_transformation_regression,
            "data_transformation_regression",
            sample_data_scenarios
        )
        
        await self.run_test(
            test_class.test_csv_loading_regression,
            "csv_loading_regression"
        )
        
        await self.run_test(
            test_class.test_error_handling_regression,
            "error_handling_regression"
        )
        
        await self.run_test(
            test_class.test_performance_regression,
            "performance_regression",
            sample_data_scenarios
        )
    
    async def validate_comprehensive_integration(self):
        """Validate comprehensive integration tests."""
        self.log("=== Comprehensive Integration Tests ===")
        
        test_class = TestComprehensiveIntegration()
        
        # Create fixtures
        mock_repository = MockRepository()
        mock_embedding_service = MockEmbeddingProvider()
        sample_csv_data = test_class.sample_csv_data()
        sample_json_data = test_class.sample_json_data()
        
        # Create temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_csv_data.to_csv(f.name, index=False)
            temp_csv = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_json_data, f)
            temp_json = f.name
        
        try:
            # Run comprehensive tests
            await self.run_test(
                test_class.test_loader_interface_compliance,
                "loader_interface_compliance"
            )
            
            await self.run_test(
                test_class.test_csv_backward_compatibility,
                "csv_backward_compatibility",
                temp_csv, sample_csv_data
            )
            
            await self.run_test(
                test_class.test_json_compatibility_and_enhancement,
                "json_compatibility_and_enhancement",
                sample_json_data
            )
            
            await self.run_test(
                test_class.test_dataload_use_case_integration,
                "dataload_use_case_integration",
                mock_repository, mock_embedding_service, temp_csv
            )
            
            await self.run_test(
                test_class.test_data_api_json_use_case_integration,
                "data_api_json_use_case_integration",
                mock_repository, mock_embedding_service, sample_json_data
            )
            
            await self.run_test(
                test_class.test_column_mapping_integration,
                "column_mapping_integration",
                mock_repository, mock_embedding_service, sample_json_data
            )
            
            await self.run_test(
                test_class.test_data_transformation_integration,
                "data_transformation_integration",
                mock_repository, mock_embedding_service, sample_json_data
            )
            
            await self.run_test(
                test_class.test_existing_table_upsert_integration,
                "existing_table_upsert_integration",
                mock_repository, mock_embedding_service, sample_json_data
            )
            
            await self.run_test(
                test_class.test_embedding_provider_compatibility,
                "embedding_provider_compatibility",
                mock_repository
            )
            
            await self.run_test(
                test_class.test_concurrent_operations,
                "concurrent_operations",
                mock_repository, mock_embedding_service
            )
            
            await self.run_test(
                test_class.test_error_handling_compatibility,
                "error_handling_compatibility",
                mock_repository, mock_embedding_service
            )
            
            await self.run_test(
                test_class.test_performance_regression,
                "performance_regression",
                mock_repository, mock_embedding_service
            )
            
            await self.run_test(
                test_class.test_migration_compatibility,
                "migration_compatibility"
            )
            
        finally:
            os.unlink(temp_csv)
            os.unlink(temp_json)
    
    async def run_all_validations(self) -> ValidationReport:
        """Run all validation tests."""
        self.log("Starting comprehensive integration validation...")
        
        # Run all validation categories
        await self.validate_storage_loader_compatibility()
        await self.validate_use_case_compatibility()
        await self.validate_embedding_provider_compatibility()
        await self.validate_database_repository_compatibility()
        await self.validate_regression_tests()
        await self.validate_comprehensive_integration()
        
        # Generate report
        total_duration = time.time() - self.start_time
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = len(self.results) - passed_tests
        
        # Calculate summary statistics
        summary = {
            "success_rate": (passed_tests / len(self.results)) * 100 if self.results else 0,
            "average_test_duration": sum(r.duration for r in self.results) / len(self.results) if self.results else 0,
            "categories": {
                "storage_loader": len([r for r in self.results if "storage_loader" in r.test_name or "csv_loading" in r.test_name or "json_loading" in r.test_name]),
                "use_case": len([r for r in self.results if "use_case" in r.test_name]),
                "embedding": len([r for r in self.results if "embedding" in r.test_name or "provider" in r.test_name]),
                "database": len([r for r in self.results if "repository" in r.test_name or "postgres" in r.test_name or "upsert" in r.test_name]),
                "regression": len([r for r in self.results if "regression" in r.test_name]),
                "integration": len([r for r in self.results if "integration" in r.test_name or "compatibility" in r.test_name])
            },
            "failed_tests": [r.test_name for r in self.results if not r.success]
        }
        
        report = ValidationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_tests=len(self.results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_duration=total_duration,
            results=self.results,
            summary=summary
        )
        
        return report
    
    def print_report(self, report: ValidationReport):
        """Print validation report to console."""
        print("\n" + "="*80)
        print("FINAL INTEGRATION VALIDATION REPORT")
        print("="*80)
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed_tests} ✅")
        print(f"Failed: {report.failed_tests} ❌")
        print(f"Success Rate: {report.summary['success_rate']:.1f}%")
        print(f"Total Duration: {report.total_duration:.2f}s")
        print(f"Average Test Duration: {report.summary['average_test_duration']:.2f}s")
        
        print("\nTest Categories:")
        for category, count in report.summary['categories'].items():
            print(f"  {category.replace('_', ' ').title()}: {count} tests")
        
        if report.failed_tests > 0:
            print(f"\nFailed Tests ({report.failed_tests}):")
            for result in report.results:
                if not result.success:
                    print(f"  ❌ {result.test_name}: {result.error_message}")
        
        print("\nDetailed Results:")
        for result in report.results:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.test_name} ({result.duration:.2f}s)")
        
        print("\n" + "="*80)
        
        if report.failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Integration validation successful.")
        else:
            print(f"⚠️  {report.failed_tests} tests failed. Review failures above.")
        
        print("="*80)


async def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Final Integration Validation for APIJSONStorageLoader")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--report-file", "-r", help="Save report to JSON file")
    
    args = parser.parse_args()
    
    # Run validation
    validator = IntegrationValidator(verbose=args.verbose)
    report = await validator.run_all_validations()
    
    # Print report
    validator.print_report(report)
    
    # Save report to file if requested
    if args.report_file:
        report_data = asdict(report)
        with open(args.report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"\nReport saved to: {args.report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if report.failed_tests == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())