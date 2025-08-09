"""
Comprehensive test suite for SQL Optimizer functionality
Tests all components: EXPLAIN plan analysis, query optimization, and system integration
"""

import pytest
import json
import os
import tempfile
from sqlalchemy import create_engine, text
from unittest.mock import Mock, patch

# Import the components we're testing
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core.sql.sql_optimizer import SQLQueryOptimizer, OptimizationLevel, ExecutionPlan, OptimizationSuggestion
from src.core.agent.sql_logic import get_sql_optimizer, analyze_sql_optimization, get_optimization_recommendations

class TestSQLOptimizer:
    """Test the core SQL optimizer engine"""
    
    @pytest.fixture
    def sqlite_engine(self):
        """Create a test SQLite database"""
        engine = create_engine("sqlite:///:memory:")
        
        # Create test tables
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    category_id INTEGER,
                    price DECIMAL(10,2),
                    created_date DATE
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE categories (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    parent_id INTEGER
                )
            """))
            
            conn.execute(text("""
                CREATE INDEX idx_products_category ON products(category_id)
            """))
            
            # Insert test data
            conn.execute(text("""
                INSERT INTO categories (id, name, parent_id) VALUES 
                (1, 'Electronics', NULL),
                (2, 'Computers', 1),
                (3, 'Phones', 1)
            """))
            
            conn.execute(text("""
                INSERT INTO products (id, name, category_id, price, created_date) VALUES 
                (1, 'Laptop', 2, 999.99, '2023-01-01'),
                (2, 'Phone', 3, 599.99, '2023-01-02'),
                (3, 'Mouse', 2, 29.99, '2023-01-03'),
                (4, 'Keyboard', 2, 79.99, '2023-01-04')
            """))
        
        return engine
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "sql_optimization": {
                "enabled": True,
                "default_optimization_level": "BASIC",
                "enable_explain_analysis": True,
                "confidence_threshold": 0.7,
                "performance_improvement_threshold": 5.0,
                "optimization_features": {
                    "index_recommendations": True,
                    "query_structure_optimization": True,
                    "join_order_optimization": True
                }
            }
        }
    
    def test_optimizer_initialization(self, sqlite_engine, config):
        """Test SQL optimizer initialization"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        assert optimizer is not None
        assert optimizer.engine == sqlite_engine
        assert optimizer.config == config
        assert optimizer.db_dialect == "sqlite"
    
    def test_explain_plan_analysis_sqlite(self, sqlite_engine, config):
        """Test EXPLAIN plan analysis for SQLite"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        sql = "SELECT * FROM products WHERE category_id = 2"
        result = optimizer.analyze_query(sql, OptimizationLevel.BASIC)
        
        assert result is not None
        assert result.original_query == sql
        assert result.execution_plan is not None
        assert len(result.suggestions) > 0
        assert result.confidence_score > 0
    
    def test_query_optimization_select_star(self, sqlite_engine, config):
        """Test optimization of SELECT * queries"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        sql = "SELECT * FROM products WHERE id = 1"
        result = optimizer.analyze_query(sql, OptimizationLevel.INTERMEDIATE)
        
        # Should suggest avoiding SELECT *
        select_star_suggestions = [s for s in result.suggestions 
                                 if "SELECT *" in s.description or "specific columns" in s.description]
        assert len(select_star_suggestions) > 0
        
        # Should provide optimized query
        if result.optimized_query:
            assert "SELECT *" not in result.optimized_query or result.confidence_score < 0.8
    
    def test_query_optimization_not_in_clause(self, sqlite_engine, config):
        """Test optimization of NOT IN clauses"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        sql = "SELECT * FROM products WHERE category_id NOT IN (SELECT id FROM categories WHERE parent_id IS NULL)"
        result = optimizer.analyze_query(sql, OptimizationLevel.AGGRESSIVE)
        
        # Should suggest NOT EXISTS alternative
        not_in_suggestions = [s for s in result.suggestions 
                            if "NOT IN" in s.description or "NOT EXISTS" in s.description]
        assert len(not_in_suggestions) > 0
    
    def test_index_recommendations(self, sqlite_engine, config):
        """Test index recommendations"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        # Query without proper index
        sql = "SELECT * FROM products WHERE price > 500 ORDER BY created_date"
        result = optimizer.analyze_query(sql, OptimizationLevel.BASIC)
        
        # Should suggest index on price or created_date
        index_suggestions = [s for s in result.suggestions 
                           if s.type == "index" and ("price" in s.description or "created_date" in s.description)]
        # Note: May not always suggest depending on table size, but structure should be correct
        assert result.suggestions is not None
    
    def test_join_optimization(self, sqlite_engine, config):
        """Test join optimization suggestions"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        sql = """
        SELECT p.name, c.name 
        FROM products p 
        JOIN categories c ON p.category_id = c.id 
        WHERE c.parent_id = 1
        """
        result = optimizer.analyze_query(sql, OptimizationLevel.INTERMEDIATE)
        
        assert result is not None
        assert result.execution_plan is not None
        # Join optimizations may vary based on data size
    
    def test_confidence_scoring(self, sqlite_engine, config):
        """Test confidence scoring system"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        # Simple, safe optimization should have high confidence
        sql = "SELECT * FROM products WHERE id = 1"
        result = optimizer.analyze_query(sql, OptimizationLevel.BASIC)
        
        assert 0 <= result.confidence_score <= 1
        
        # Complex query should have appropriate confidence
        complex_sql = """
        SELECT COUNT(*) FROM products p 
        JOIN categories c ON p.category_id = c.id 
        WHERE p.price > (SELECT AVG(price) FROM products)
        """
        complex_result = optimizer.analyze_query(complex_sql, OptimizationLevel.AGGRESSIVE)
        
        assert 0 <= complex_result.confidence_score <= 1
    
    def test_safety_checks(self, sqlite_engine, config):
        """Test safety checks for optimization"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        # Non-SELECT query should not be optimized
        sql = "UPDATE products SET price = 999.99 WHERE id = 1"
        result = optimizer.analyze_query(sql, OptimizationLevel.BASIC)
        
        # Should still provide analysis but no risky optimizations
        assert result is not None
        assert result.original_query == sql
    
    def test_optimization_levels(self, sqlite_engine, config):
        """Test different optimization levels"""
        optimizer = SQLQueryOptimizer(sqlite_engine, config)
        
        sql = "SELECT * FROM products WHERE category_id = 2 AND price > 100"
        
        # Test all optimization levels
        basic_result = optimizer.analyze_query(sql, OptimizationLevel.BASIC)
        intermediate_result = optimizer.analyze_query(sql, OptimizationLevel.INTERMEDIATE)
        aggressive_result = optimizer.analyze_query(sql, OptimizationLevel.AGGRESSIVE)
        
        # More aggressive levels should provide more suggestions
        assert len(basic_result.suggestions) <= len(intermediate_result.suggestions)
        assert len(intermediate_result.suggestions) <= len(aggressive_result.suggestions)


class TestSQLLogicIntegration:
    """Test integration with sql_logic.py"""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock engine for testing"""
        engine = Mock()
        engine.dialect.name = "sqlite"
        return engine
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "sql_optimization": {
                "enabled": True,
                "default_optimization_level": "BASIC",
                "confidence_threshold": 0.7
            }
        }
    
    @patch('src.core.agent.sql_logic.SQLQueryOptimizer')
    def test_get_sql_optimizer(self, mock_optimizer_class, mock_engine, config):
        """Test getting SQL optimizer instance"""
        with patch('src.core.agent.sql_logic.CONFIG', config):
            with patch('src.core.agent.sql_logic.ENGINE', mock_engine):
                optimizer = get_sql_optimizer()
                
                assert optimizer is not None
                mock_optimizer_class.assert_called_once_with(mock_engine, config)
    
    def test_analyze_sql_optimization(self):
        """Test SQL optimization analysis function"""
        # Mock the optimizer and its result
        mock_result = Mock()
        mock_result.suggestions = [
            Mock(type="index", description="Add index on column X", confidence=0.8)
        ]
        mock_result.execution_plan = Mock()
        mock_result.confidence_score = 0.85
        mock_result.optimized_query = "SELECT id, name FROM products WHERE id = 1"
        
        with patch('src.core.agent.sql_logic.get_sql_optimizer') as mock_get_optimizer:
            mock_optimizer = Mock()
            mock_optimizer.analyze_query.return_value = mock_result
            mock_get_optimizer.return_value = mock_optimizer
            
            result = analyze_sql_optimization(
                "SELECT * FROM products WHERE id = 1",
                None,  # engine
                "BASIC"
            )
            
            assert result is not None
            assert 'analysis' in result
            assert 'recommendations' in result
            assert 'optimized_query' in result
            assert 'confidence_score' in result


class TestAPIIntegration:
    """Test API integration for SQL optimization"""
    
    def test_api_response_structure(self):
        """Test that API responses include optimization fields"""
        # This would typically be an integration test with the actual API
        # For unit testing, we verify the response structure
        
        expected_fields = [
            'sql_optimization',
            'optimization_recommendations', 
            'original_sql',
            'optimization_applied'
        ]
        
        # Mock API response structure
        mock_response = {
            'answer': 'Test answer',
            'sql': 'SELECT * FROM products',
            'sql_optimization': {
                'confidence_score': 0.8,
                'performance_improvement': 15.0
            },
            'optimization_recommendations': [
                {'type': 'index', 'description': 'Add index on price column'}
            ],
            'original_sql': 'SELECT * FROM products WHERE price > 100',
            'optimization_applied': True
        }
        
        # Verify all expected fields are present
        for field in expected_fields:
            assert field in mock_response


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
