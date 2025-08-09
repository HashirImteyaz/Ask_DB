# run_comprehensive_test.py - Run comprehensive NLQ testing

import os
import time
import json
import pandas as pd
import subprocess
import threading
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveNLQTester:
    """Comprehensive testing suite for NLQ system using evaluation questions."""
    
    def __init__(self, server_port=8000):
        self.server_port = server_port
        self.api_base_url = f"http://127.0.0.1:{server_port}"
        self.server_process = None
        self.test_results = []
        
    def start_server_and_setup(self):
        """Start server and upload schema."""
        print("Starting NLQ server...")
        
        # Start server in background
        try:
            self.server_process = subprocess.Popen(
                ["python", "main_app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait for server to start
            for i in range(30):
                try:
                    response = requests.get(f"{self.api_base_url}/docs", timeout=2)
                    if response.status_code == 200:
                        print(f"[PASS] Server started successfully on port {self.server_port}")
                        break
                except requests.exceptions.RequestException:
                    time.sleep(1)
            else:
                raise Exception("Server failed to start within 30 seconds")
            
            # Upload schema description
            self.upload_schema()
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def upload_schema(self):
        """Upload schema description to initialize retrievers."""
        schema_file = "src/config/schema_description.json"
        
        if not Path(schema_file).exists():
            raise Exception(f"Schema file not found: {schema_file}")
        
        try:
            with open(schema_file, 'rb') as f:
                files = {'context_file': f}
                response = requests.post(f"{self.api_base_url}/upload", files=files, timeout=30)
            
            if response.status_code == 200:
                print("[PASS] Schema description uploaded successfully")
                time.sleep(2)  # Allow time for retriever initialization
            else:
                raise Exception(f"Schema upload failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Error uploading schema: {e}")
    
    def query_nlq_system(self, question: str):
        """Query the NLQ system and return structured response."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_base_url}/chat",
                json={"query": question, "history": []},
                timeout=45,
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
                    'raw_response': data
                }
            else:
                return {
                    'success': False,
                    'response_time_ms': response_time,
                    'sql': None,
                    'answer': response.text,
                    'error': f"HTTP {response.status_code}",
                    'raw_response': None
                }
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'success': False,
                'response_time_ms': response_time,
                'sql': None,
                'answer': '',
                'error': str(e),
                'raw_response': None
            }
    
    def analyze_sql_similarity(self, generated_sql, expected_sql):
        """Analyze similarity between generated and expected SQL."""
        if not generated_sql or not expected_sql:
            return {
                'similarity_score': 0.0,
                'analysis': 'Missing SQL query',
                'tables_match': False,
                'operations_match': False
            }
        
        gen_upper = generated_sql.upper()
        exp_upper = expected_sql.upper()
        
        # Extract components
        gen_tables = self._extract_table_names(gen_upper)
        exp_tables = self._extract_table_names(exp_upper)
        
        gen_operations = self._extract_sql_operations(gen_upper)
        exp_operations = self._extract_sql_operations(exp_upper)
        
        # Calculate matches
        table_overlap = len(set(gen_tables) & set(exp_tables))
        operation_overlap = len(set(gen_operations) & set(exp_operations))
        
        tables_match = table_overlap > 0
        operations_match = operation_overlap > 0
        
        # Calculate similarity score
        score = 0.0
        if tables_match:
            score += 0.4 * (table_overlap / max(len(exp_tables), 1))
        if operations_match:
            score += 0.4 * (operation_overlap / max(len(exp_operations), 1))
        
        # Bonus for structure similarity
        if 'JOIN' in gen_upper and 'JOIN' in exp_upper:
            score += 0.1
        if 'WHERE' in gen_upper and 'WHERE' in exp_upper:
            score += 0.1
        
        return {
            'similarity_score': min(score, 1.0),
            'analysis': f"Tables: {gen_tables} vs {exp_tables}, Ops: {gen_operations} vs {exp_operations}",
            'tables_match': tables_match,
            'operations_match': operations_match,
            'generated_tables': gen_tables,
            'expected_tables': exp_tables,
            'generated_operations': gen_operations,
            'expected_operations': exp_operations
        }
    
    def _extract_table_names(self, sql):
        """Extract table names from SQL."""
        tables = []
        if 'SPECIFICATIONS' in sql:
            tables.append('Specifications')
        if 'RECIPEEXPLOSION' in sql:
            tables.append('RecipeExplosion')
        return tables
    
    def _extract_sql_operations(self, sql):
        """Extract SQL operations."""
        operations = []
        for op in ['SELECT', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY', 'COUNT', 'SUM', 'DISTINCT', 'LIKE']:
            if op in sql:
                operations.append(op)
        return operations
    
    def run_evaluation_test(self, question, expected_sql, difficulty, test_id):
        """Run a single evaluation test."""
        print(f"Testing [{difficulty}]: {question}")
        
        # Query system
        result = self.query_nlq_system(question)
        
        # Analyze SQL similarity
        sql_analysis = self.analyze_sql_similarity(result['sql'], expected_sql)
        
        # Determine success criteria
        sql_generated = bool(result['sql'])
        answer_provided = bool(result['answer'])
        sql_reasonable = sql_analysis['similarity_score'] > 0.3  # At least 30% similarity
        
        overall_success = (
            result['success'] and 
            sql_generated and 
            answer_provided and
            sql_reasonable
        )
        
        test_result = {
            'test_id': test_id,
            'difficulty': difficulty,
            'question': question,
            'expected_sql': expected_sql,
            'generated_sql': result['sql'],
            'answer': result['answer'],
            'response_time_ms': result['response_time_ms'],
            'api_success': result['success'],
            'sql_generated': sql_generated,
            'answer_provided': answer_provided,
            'sql_similarity_score': sql_analysis['similarity_score'],
            'sql_analysis': sql_analysis,
            'overall_success': overall_success,
            'error': result['error']
        }
        
        # Print result
        if overall_success:
            print(f"  [PASS] Score: {sql_analysis['similarity_score']:.2f}, Time: {result['response_time_ms']:.0f}ms")
        else:
            reason = result['error'] or "Low SQL similarity" if not sql_reasonable else "Missing SQL/Answer"
            print(f"  [FAIL] Reason: {reason}")
        
        self.test_results.append(test_result)
        return test_result
    
    def run_all_tests(self):
        """Run all evaluation tests from the Excel file."""
        print("="*70)
        print("COMPREHENSIVE NLQ SYSTEM EVALUATION")
        print("="*70)
        
        # Load evaluation questions
        try:
            eval_df = pd.read_excel("evaluation/evaluation_questions.xlsx")
            print(f"Loaded {len(eval_df)} evaluation questions")
        except Exception as e:
            logger.error(f"Failed to load evaluation questions: {e}")
            return None
        
        # Start server
        if not self.start_server_and_setup():
            logger.error("Failed to start server. Aborting tests.")
            return None
        
        print(f"\nRunning {len(eval_df)} evaluation tests...\n")
        
        # Run tests
        for idx, row in eval_df.iterrows():
            if pd.isna(row['Human Question']):  # Skip empty questions
                continue
                
            self.run_evaluation_test(
                question=row['Human Question'],
                expected_sql=row['SQL Query'],
                difficulty=row.get('Difficulty', 'Unknown'),
                test_id=f"eval_{idx+1:02d}"
            )
            
            # Small delay between tests
            time.sleep(0.5)
        
        # Calculate summary statistics
        summary = self.calculate_summary()
        return summary
    
    def calculate_summary(self):
        """Calculate comprehensive summary statistics."""
        if not self.test_results:
            return None
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['overall_success'])
        
        # Calculate averages
        avg_response_time = sum(r['response_time_ms'] for r in self.test_results) / total_tests
        avg_sql_score = sum(r['sql_similarity_score'] for r in self.test_results) / total_tests
        
        # Breakdown by difficulty
        difficulty_stats = {}
        for result in self.test_results:
            diff = result['difficulty']
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'total': 0, 'passed': 0}
            difficulty_stats[diff]['total'] += 1
            if result['overall_success']:
                difficulty_stats[diff]['passed'] += 1
        
        # Component success rates
        sql_generated = sum(1 for r in self.test_results if r['sql_generated'])
        answers_provided = sum(1 for r in self.test_results if r['answer_provided'])
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests,
            'avg_response_time_ms': avg_response_time,
            'avg_sql_similarity_score': avg_sql_score,
            'sql_generation_rate': sql_generated / total_tests,
            'answer_provision_rate': answers_provided / total_tests,
            'difficulty_breakdown': {
                diff: {
                    'success_rate': stats['passed'] / stats['total'],
                    'passed': stats['passed'],
                    'total': stats['total']
                }
                for diff, stats in difficulty_stats.items()
            },
            'individual_results': self.test_results
        }
        
        return summary
    
    def print_summary(self, summary):
        """Print formatted summary."""
        if not summary:
            print("No test results to summarize.")
            return
        
        print("\n" + "="*70)
        print("EVALUATION RESULTS SUMMARY")
        print("="*70)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Overall Success Rate: {summary['success_rate']:.1%}")
        print(f"Average Response Time: {summary['avg_response_time_ms']:.0f}ms")
        print(f"Average SQL Similarity: {summary['avg_sql_similarity_score']:.2f}")
        print(f"SQL Generation Rate: {summary['sql_generation_rate']:.1%}")
        print(f"Answer Provision Rate: {summary['answer_provision_rate']:.1%}")
        
        print(f"\nBREAKDOWN BY DIFFICULTY:")
        for difficulty, stats in summary['difficulty_breakdown'].items():
            print(f"  {difficulty}: {stats['passed']}/{stats['total']} ({stats['success_rate']:.1%})")
        
        print(f"\nFAILED TESTS:")
        failed_tests = [r for r in summary['individual_results'] if not r['overall_success']]
        for result in failed_tests:
            reason = result['error'] or f"Low similarity ({result['sql_similarity_score']:.2f})"
            print(f"  [{result['difficulty']}] {result['test_id']}: {reason}")
        
        # Save detailed results
        output_file = f"evaluation/test_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    
    def cleanup(self):
        """Clean up server process."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            print("\nServer stopped.")

def main():
    """Main testing function."""
    tester = ComprehensiveNLQTester()
    
    try:
        summary = tester.run_all_tests()
        if summary:
            tester.print_summary(summary)
            
            # Return success based on performance threshold
            success_rate = summary['success_rate']
            print(f"\nFINAL RESULT: {success_rate:.1%} success rate")
            
            if success_rate >= 0.7:
                print("EXCELLENT: System performance exceeds 70% threshold!")
                return True
            elif success_rate >= 0.5:
                print("GOOD: System performance is acceptable (50%+ threshold)")
                return True
            else:
                print("NEEDS IMPROVEMENT: System performance below 50% threshold")
                return False
        else:
            print("Testing failed - no results generated")
            return False
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Testing failed with exception: {e}")
        return False
    finally:
        tester.cleanup()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)