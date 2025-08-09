# evaluation/nlq_evaluator.py - NLQ System Testing and Evaluation Framework

import json
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sqlalchemy import create_engine, text
import requests
import logging
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a single NLQ test case."""
    id: str
    question: str
    expected_result_type: str  # 'count', 'list', 'calculation', 'boolean'
    expected_sql_patterns: List[str]  # SQL patterns that should appear in the query
    category: str  # 'basic', 'joins', 'aggregations', 'business_logic'
    description: str
    expected_tables: List[str]  # Tables that should be queried
    
@dataclass
class EvaluationResult:
    """Results of evaluating a single test case."""
    test_case_id: str
    success: bool
    response_time_ms: float
    sql_generated: Optional[str]
    answer_provided: str
    error_message: Optional[str]
    sql_correctness_score: float  # 0.0 to 1.0
    relevance_score: float  # 0.0 to 1.0

class NLQEvaluator:
    """Evaluates NLQ system accuracy and performance."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", db_url: str = None):
        self.api_base_url = api_base_url
        self.db_url = db_url
        self.engine = create_engine(db_url) if db_url else None
        
    def load_test_cases(self, test_file_path: str) -> List[TestCase]:
        """Load test cases from JSON file."""
        with open(test_file_path, 'r') as f:
            data = json.load(f)
        
        test_cases = []
        for item in data['test_cases']:
            test_cases.append(TestCase(**item))
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def evaluate_sql_correctness(self, generated_sql: str, expected_patterns: List[str], expected_tables: List[str]) -> float:
        """Evaluate SQL query correctness based on expected patterns and tables."""
        if not generated_sql:
            return 0.0
        
        score = 0.0
        total_checks = len(expected_patterns) + len(expected_tables)
        
        # Check for expected SQL patterns
        sql_upper = generated_sql.upper()
        for pattern in expected_patterns:
            if pattern.upper() in sql_upper:
                score += 1.0
        
        # Check for expected tables
        for table in expected_tables:
            if table.upper() in sql_upper:
                score += 1.0
        
        return score / total_checks if total_checks > 0 else 0.0
    
    def evaluate_sql_execution(self, sql: str) -> Tuple[bool, Optional[str]]:
        """Test if SQL query executes without errors."""
        if not self.engine:
            return True, "No database connection for execution testing"
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                # Just test execution, don't fetch all results
                result.fetchone()
            return True, None
        except Exception as e:
            return False, str(e)
    
    def query_nlq_system(self, question: str) -> Tuple[Optional[str], str, float, Optional[str]]:
        """Query the NLQ system and return SQL, answer, response time, and any error."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_base_url}/chat",
                json={
                    "query": question,
                    "history": []
                },
                timeout=30
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return data.get('sql'), data.get('answer', ''), response_time, None
            else:
                return None, f"HTTP {response.status_code}", response_time, response.text
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return None, "Request failed", response_time, str(e)
    
    def evaluate_answer_relevance(self, question: str, answer: str, expected_type: str) -> float:
        """Evaluate answer relevance (simplified heuristic-based evaluation)."""
        if not answer or answer.strip() == "":
            return 0.0
        
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        score = 0.0
        
        # Check if answer type matches expected
        if expected_type == 'count' and any(word in answer_lower for word in ['total', 'count', 'number', 'are']):
            score += 0.3
        elif expected_type == 'list' and ('•' in answer or '1.' in answer or 'include' in answer_lower):
            score += 0.3
        elif expected_type == 'calculation' and any(word in answer_lower for word in ['sum', 'total', 'amount', 'quantity']):
            score += 0.3
        elif expected_type == 'boolean' and any(word in answer_lower for word in ['yes', 'no', 'true', 'false']):
            score += 0.3
        
        # Check if key terms from question appear in answer
        question_words = set(question_lower.split())
        answer_words = set(answer_lower.split())
        overlap = question_words.intersection(answer_words)
        
        if len(question_words) > 0:
            term_overlap_score = len(overlap) / len(question_words)
            score += min(term_overlap_score, 0.4)
        
        # Check answer length (not too short, not too long)
        if 20 <= len(answer) <= 500:
            score += 0.3
        
        return min(score, 1.0)
    
    def evaluate_test_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case."""
        logger.info(f"Evaluating: {test_case.id} - {test_case.question}")
        
        sql, answer, response_time, error = self.query_nlq_system(test_case.question)
        
        if error:
            return EvaluationResult(
                test_case_id=test_case.id,
                success=False,
                response_time_ms=response_time,
                sql_generated=sql,
                answer_provided=answer,
                error_message=error,
                sql_correctness_score=0.0,
                relevance_score=0.0
            )
        
        # Evaluate SQL correctness
        sql_score = self.evaluate_sql_correctness(
            sql or "", 
            test_case.expected_sql_patterns, 
            test_case.expected_tables
        )
        
        # Test SQL execution if possible
        execution_success = True
        if sql and self.engine:
            execution_success, exec_error = self.evaluate_sql_execution(sql)
            if not execution_success:
                logger.warning(f"SQL execution failed for {test_case.id}: {exec_error}")
        
        # Evaluate answer relevance
        relevance_score = self.evaluate_answer_relevance(
            test_case.question, 
            answer, 
            test_case.expected_result_type
        )
        
        success = sql_score > 0.5 and relevance_score > 0.5 and execution_success
        
        return EvaluationResult(
            test_case_id=test_case.id,
            success=success,
            response_time_ms=response_time,
            sql_generated=sql,
            answer_provided=answer,
            error_message=None,
            sql_correctness_score=sql_score,
            relevance_score=relevance_score
        )
    
    def run_evaluation(self, test_cases: List[TestCase]) -> Dict:
        """Run evaluation on all test cases and return comprehensive results."""
        results = []
        category_stats = {}
        
        logger.info(f"Starting evaluation of {len(test_cases)} test cases...")
        
        for test_case in test_cases:
            result = self.evaluate_test_case(test_case)
            results.append(result)
            
            # Track category statistics
            if test_case.category not in category_stats:
                category_stats[test_case.category] = {'total': 0, 'passed': 0}
            
            category_stats[test_case.category]['total'] += 1
            if result.success:
                category_stats[test_case.category]['passed'] += 1
        
        # Calculate overall statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        avg_response_time = sum(r.response_time_ms for r in results) / total_tests
        avg_sql_score = sum(r.sql_correctness_score for r in results) / total_tests
        avg_relevance_score = sum(r.relevance_score for r in results) / total_tests
        
        # Prepare summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests,
            'avg_response_time_ms': avg_response_time,
            'avg_sql_correctness_score': avg_sql_score,
            'avg_relevance_score': avg_relevance_score,
            'category_breakdown': {
                cat: {
                    'success_rate': stats['passed'] / stats['total'],
                    'passed': stats['passed'],
                    'total': stats['total']
                }
                for cat, stats in category_stats.items()
            },
            'individual_results': [
                {
                    'test_id': r.test_case_id,
                    'success': r.success,
                    'response_time_ms': r.response_time_ms,
                    'sql_correctness_score': r.sql_correctness_score,
                    'relevance_score': r.relevance_score,
                    'sql_generated': r.sql_generated,
                    'answer_provided': r.answer_provided[:200] + '...' if len(r.answer_provided) > 200 else r.answer_provided,
                    'error_message': r.error_message
                }
                for r in results
            ]
        }
        
        return summary
    
    def save_results(self, results: Dict, output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

def create_sample_test_cases():
    """Create sample test cases for PLM data."""
    test_cases = {
        "test_cases": [
            {
                "id": "basic_001",
                "question": "How many specifications are there?",
                "expected_result_type": "count",
                "expected_sql_patterns": ["COUNT", "Specifications"],
                "category": "basic",
                "description": "Basic count query",
                "expected_tables": ["Specifications"]
            },
            {
                "id": "basic_002", 
                "question": "What are the different business units?",
                "expected_result_type": "list",
                "expected_sql_patterns": ["DISTINCT", "CUCPlantBUName"],
                "category": "basic",
                "description": "List distinct values",
                "expected_tables": ["RecipeExplosion"]
            },
            {
                "id": "joins_001",
                "question": "What ingredients are used in food products?",
                "expected_result_type": "list", 
                "expected_sql_patterns": ["JOIN", "SpecGroupCode", "ZFOD"],
                "category": "joins",
                "description": "Join between tables with filtering",
                "expected_tables": ["Specifications", "RecipeExplosion"]
            },
            {
                "id": "aggregation_001",
                "question": "What is the total percentage contribution of salt across all recipes?",
                "expected_result_type": "calculation",
                "expected_sql_patterns": ["SUM", "Ing2CUC_PercentageContribution"],
                "category": "aggregations", 
                "description": "Aggregation with filtering",
                "expected_tables": ["Specifications", "RecipeExplosion"]
            }
        ]
    }
    
    return test_cases

if __name__ == "__main__":
    # Create sample test cases
    sample_tests = create_sample_test_cases()
    with open("evaluation/sample_test_cases.json", "w") as f:
        json.dump(sample_tests, f, indent=2)
    
    # Run evaluation
    evaluator = NLQEvaluator(
        api_base_url="http://localhost:8000",
        db_url="sqlite:///DATA/plm_updated.db"
    )
    
    test_cases = evaluator.load_test_cases("evaluation/sample_test_cases.json")
    results = evaluator.run_evaluation(test_cases)
    evaluator.save_results(results, f"evaluation/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Average Response Time: {results['avg_response_time_ms']:.0f}ms")
    print(f"Average SQL Score: {results['avg_sql_correctness_score']:.2f}")
    print(f"Average Relevance Score: {results['avg_relevance_score']:.2f}")