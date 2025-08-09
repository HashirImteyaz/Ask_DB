# test_nlq_system.py - Comprehensive NLQ System Testing

import pandas as pd
import requests
import json
import time
import logging
from pathlib import Path
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NLQSystemTester:
    """Comprehensive testing suite for the NLQ system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.db_url = os.getenv('DB_URL', 'sqlite:///DATA/plm_updated.db')
        self.engine = create_engine(self.db_url)
        self.test_results = []
        
    def load_evaluation_questions(self, excel_file: str) -> pd.DataFrame:
        """Load evaluation questions from Excel file."""
        try:
            df = pd.read_excel(excel_file)
            logger.info(f"Loaded {len(df)} evaluation questions from {excel_file}")
            
            # Display the columns and first few rows
            logger.info(f"Columns: {list(df.columns)}")
            logger.info("First 3 questions:")
            for i, row in df.head(3).iterrows():
                logger.info(f"  Q{i+1}: {row.iloc[0]}")
                if len(row) > 1:
                    logger.info(f"       Expected SQL: {str(row.iloc[1])[:100]}...")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise
    
    def test_database_connection(self) -> bool:
        """Test database connectivity and basic structure."""
        logger.info("Testing database connection...")
        try:
            with self.engine.connect() as conn:
                # Check if required tables exist
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                tables = [row[0] for row in result]
                logger.info(f"Available tables: {tables}")
                
                required_tables = ['Specifications', 'RecipeExplosion']
                missing_tables = [table for table in required_tables if table not in tables]
                
                if missing_tables:
                    logger.error(f"Missing required tables: {missing_tables}")
                    return False
                
                # Check table contents
                for table in required_tables:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    logger.info(f"Table {table}: {count} records")
                
                logger.info("✅ Database connection test passed")
                return True
                
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def test_api_health(self) -> bool:
        """Test API server health and availability."""
        logger.info("Testing API health...")
        try:
            response = requests.get(f"{self.api_base_url}/docs", timeout=10)
            if response.status_code == 200:
                logger.info("✅ API health test passed")
                return True
            else:
                logger.error(f"❌ API returned status code: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API health test failed: {e}")
            return False
    
    def query_nlq_api(self, question: str) -> Dict:
        """Query the NLQ API and return response details."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_base_url}/chat",
                json={
                    "query": question,
                    "history": []
                },
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'response_time_ms': response_time,
                    'sql': data.get('sql'),
                    'answer': data.get('answer', ''),
                    'error': None,
                    'status_code': response.status_code
                }
            else:
                return {
                    'success': False,
                    'response_time_ms': response_time,
                    'sql': None,
                    'answer': response.text,
                    'error': f"HTTP {response.status_code}",
                    'status_code': response.status_code
                }
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'response_time_ms': response_time,
                'sql': None,
                'answer': '',
                'error': str(e),
                'status_code': None
            }
    
    def validate_sql_syntax(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Validate SQL syntax by attempting to explain the query."""
        if not sql:
            return False, "No SQL generated"
        
        try:
            with self.engine.connect() as conn:
                # Try to explain the query (won't execute, just validate syntax)
                explain_sql = f"EXPLAIN QUERY PLAN {sql}"
                conn.execute(text(explain_sql))
                return True, None
        except Exception as e:
            return False, str(e)
    
    def compare_sql_queries(self, generated_sql: str, expected_sql: str) -> Dict:
        """Compare generated SQL with expected SQL."""
        if not generated_sql or not expected_sql:
            return {
                'similarity_score': 0.0,
                'table_match': False,
                'operation_match': False,
                'notes': 'One or both SQL queries are empty'
            }
        
        gen_upper = generated_sql.upper()
        exp_upper = expected_sql.upper()
        
        # Extract key components
        gen_tables = self._extract_tables(gen_upper)
        exp_tables = self._extract_tables(exp_upper)
        
        gen_operations = self._extract_operations(gen_upper)
        exp_operations = self._extract_operations(exp_upper)
        
        # Calculate similarity
        table_match = len(set(gen_tables) & set(exp_tables)) > 0
        operation_match = len(set(gen_operations) & set(exp_operations)) > 0
        
        # Simple similarity score
        similarity_score = 0.0
        if table_match:
            similarity_score += 0.4
        if operation_match:
            similarity_score += 0.4
        if 'JOIN' in gen_upper and 'JOIN' in exp_upper:
            similarity_score += 0.2
        
        return {
            'similarity_score': similarity_score,
            'table_match': table_match,
            'operation_match': operation_match,
            'generated_tables': gen_tables,
            'expected_tables': exp_tables,
            'generated_operations': gen_operations,
            'expected_operations': exp_operations,
            'notes': f"Tables: {gen_tables} vs {exp_tables}, Operations: {gen_operations} vs {exp_operations}"
        }
    
    def _extract_tables(self, sql: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = []
        if 'SPECIFICATIONS' in sql:
            tables.append('Specifications')
        if 'RECIPEEXPLOSION' in sql:
            tables.append('RecipeExplosion')
        return tables
    
    def _extract_operations(self, sql: str) -> List[str]:
        """Extract SQL operations from query."""
        operations = []
        for op in ['SELECT', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY', 'COUNT', 'SUM', 'DISTINCT']:
            if op in sql:
                operations.append(op)
        return operations
    
    def run_single_test(self, question: str, expected_sql: str = None, test_id: str = None) -> Dict:
        """Run a single test case."""
        logger.info(f"Testing: {question}")
        
        # Query the API
        api_result = self.query_nlq_api(question)
        
        # Initialize test result
        test_result = {
            'test_id': test_id or f"test_{len(self.test_results)+1}",
            'question': question,
            'expected_sql': expected_sql,
            'api_success': api_result['success'],
            'response_time_ms': api_result['response_time_ms'],
            'generated_sql': api_result['sql'],
            'answer': api_result['answer'],
            'api_error': api_result['error'],
            'sql_valid': False,
            'sql_validation_error': None,
            'sql_comparison': None,
            'overall_success': False
        }
        
        # Validate SQL if generated
        if api_result['sql']:
            sql_valid, sql_error = self.validate_sql_syntax(api_result['sql'])
            test_result['sql_valid'] = sql_valid
            test_result['sql_validation_error'] = sql_error
            
            # Compare with expected SQL if provided
            if expected_sql:
                comparison = self.compare_sql_queries(api_result['sql'], expected_sql)
                test_result['sql_comparison'] = comparison
        
        # Determine overall success
        test_result['overall_success'] = (
            api_result['success'] and 
            bool(api_result['sql']) and 
            test_result['sql_valid'] and
            bool(api_result['answer'])
        )
        
        self.test_results.append(test_result)
        return test_result
    
    def run_evaluation_tests(self, excel_file: str) -> Dict:
        """Run all tests from the evaluation Excel file."""
        logger.info("Starting comprehensive evaluation tests...")
        
        # Load evaluation questions
        try:
            df = self.load_evaluation_questions(excel_file)
        except Exception as e:
            logger.error(f"Failed to load evaluation file: {e}")
            return {'error': str(e)}
        
        # Run tests
        total_tests = len(df)
        passed_tests = 0
        
        for idx, row in df.iterrows():
            question = row.iloc[0]  # First column: Human Question
            expected_sql = row.iloc[1] if len(row) > 1 and pd.notna(row.iloc[1]) else None
            
            test_result = self.run_single_test(
                question=question,
                expected_sql=expected_sql,
                test_id=f"eval_{idx+1}"
            )
            
            if test_result['overall_success']:
                passed_tests += 1
                logger.info(f"✅ Test {idx+1}/{total_tests} PASSED")
            else:
                logger.warning(f"❌ Test {idx+1}/{total_tests} FAILED: {test_result.get('api_error', 'Unknown error')}")
        
        # Calculate statistics
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        avg_response_time = sum(r['response_time_ms'] for r in self.test_results) / len(self.test_results)
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': success_rate,
            'avg_response_time_ms': avg_response_time,
            'individual_results': self.test_results
        }
        
        return summary
    
    def save_test_results(self, results: Dict, output_file: str):
        """Save test results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Test results saved to {output_file}")
    
    def print_summary(self, results: Dict):
        """Print a formatted summary of test results."""
        print("\n" + "="*60)
        print("NLQ SYSTEM TEST RESULTS SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Average Response Time: {results['avg_response_time_ms']:.0f}ms")
        
        print("\n📊 DETAILED BREAKDOWN:")
        
        # SQL generation success
        sql_generated = sum(1 for r in results['individual_results'] if r['generated_sql'])
        print(f"SQL Generated: {sql_generated}/{results['total_tests']} ({sql_generated/results['total_tests']:.1%})")
        
        # SQL validation success
        sql_valid = sum(1 for r in results['individual_results'] if r['sql_valid'])
        print(f"Valid SQL: {sql_valid}/{results['total_tests']} ({sql_valid/results['total_tests']:.1%})")
        
        # Answer generation success
        answers_provided = sum(1 for r in results['individual_results'] if r['answer'])
        print(f"Answers Provided: {answers_provided}/{results['total_tests']} ({answers_provided/results['total_tests']:.1%})")
        
        print("\n🔍 FAILED TESTS:")
        for result in results['individual_results']:
            if not result['overall_success']:
                print(f"❌ {result['test_id']}: {result['question'][:60]}...")
                if result['api_error']:
                    print(f"   Error: {result['api_error']}")
                elif result['sql_validation_error']:
                    print(f"   SQL Error: {result['sql_validation_error']}")
                else:
                    print(f"   Issue: No SQL or answer generated")
        
        print("\n✅ SAMPLE SUCCESSFUL TESTS:")
        successful_tests = [r for r in results['individual_results'] if r['overall_success']]
        for result in successful_tests[:3]:
            print(f"✅ {result['test_id']}: {result['question'][:60]}...")
            print(f"   Response Time: {result['response_time_ms']:.0f}ms")
            if result['generated_sql']:
                print(f"   SQL: {result['generated_sql'][:80]}...")

def main():
    """Main testing function."""
    # Initialize tester
    tester = NLQSystemTester()
    
    # Test system components
    logger.info("Running system health checks...")
    
    db_ok = tester.test_database_connection()
    if not db_ok:
        logger.error("Database test failed. Please ensure database is set up correctly.")
        return False
    
    api_ok = tester.test_api_health()
    if not api_ok:
        logger.error("API test failed. Please ensure the server is running.")
        logger.info("To start the server, run: python main_app.py")
        return False
    
    # Run evaluation tests
    excel_file = "evaluation/evaluation_questions.xlsx"
    if not Path(excel_file).exists():
        logger.error(f"Evaluation file not found: {excel_file}")
        return False
    
    results = tester.run_evaluation_tests(excel_file)
    
    if 'error' in results:
        logger.error(f"Evaluation failed: {results['error']}")
        return False
    
    # Save and display results
    output_file = f"evaluation/test_results_{int(time.time())}.json"
    tester.save_test_results(results, output_file)
    tester.print_summary(results)
    
    logger.info(f"\n🎯 Overall Success Rate: {results['success_rate']:.1%}")
    return results['success_rate'] > 0.7  # Consider 70%+ success rate as good

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)