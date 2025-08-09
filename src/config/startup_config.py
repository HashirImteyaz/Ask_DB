"""
Enhanced Startup Configuration for NLQ PLM System
Handles all configuration loading and environment setup for scalability features
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StartupConfig:
    """
    Comprehensive configuration manager for the NLQ PLM system
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the startup configuration
        
        Args:
            config_path: Path to the config.json file
        """
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self._validate_config()
        
    def _find_config_file(self) -> str:
        """Find the config.json file in the project"""
        # Start from current directory and work up
        current_dir = Path.cwd()
        
        # Check common locations
        config_locations = [
            current_dir / "config.json",
            current_dir.parent / "config.json",
            current_dir.parent.parent / "config.json",
            Path(__file__).parent.parent.parent / "config.json"
        ]
        
        for config_path in config_locations:
            if config_path.exists():
                return str(config_path)
        
        raise FileNotFoundError("config.json not found in any expected location")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate_config(self):
        """Validate critical configuration parameters"""
        required_sections = [
            "database", "llm_settings", "agent", 
            "cost_optimization", "scalability"
        ]
        
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing configuration section: {section}")
    
    # Database Configuration
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.config.get("database", {})
    
    @property
    def database_type(self) -> str:
        """Get database type (sqlite, postgresql, mysql)"""
        return self.database_config.get("type", "sqlite")
    
    @property
    def database_url(self) -> str:
        """Get database connection URL"""
        # First, check for environment variable
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url
            
        # Second, check config for direct URLs
        db_config = self.database_config
        
        if "default_url" in db_config:
            return db_config["default_url"]
        elif "postgresql_url" in db_config:
            return db_config["postgresql_url"]
        elif "mysql_url" in db_config:
            return db_config["mysql_url"]
        elif "sqlite_fallback" in db_config:
            return db_config["sqlite_fallback"]
        
        # Fallback: construct from individual components
        db_type = self.database_type
        
        if db_type == "sqlite":
            return f"sqlite:///{db_config.get('path', 'DATA/plm_updated.db')}"
        elif db_type == "postgresql":
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 5432)
            user = db_config.get("user", "postgres")
            password = db_config.get("password", "")
            database = db_config.get("database", "nlq_plm")
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        elif db_type == "mysql":
            host = db_config.get("host", "localhost")
            port = db_config.get("port", 3306)
            user = db_config.get("user", "root")
            password = db_config.get("password", "")
            database = db_config.get("database", "nlq_plm")
            return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        else:
            # Final fallback
            return "sqlite:///DATA/plm_updated.db"
    
    # Scalability Configuration
    @property
    def scalability_config(self) -> Dict[str, Any]:
        """Get scalability configuration"""
        return self.config.get("scalability", {})
    
    @property
    def async_processing_enabled(self) -> bool:
        """Check if async processing is enabled"""
        return self.scalability_config.get("async_processing", {}).get("enabled", False)
    
    @property
    def caching_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self.scalability_config.get("caching", {}).get("enabled", False)
    
    @property
    def sql_validation_enabled(self) -> bool:
        """Check if SQL validation is enabled"""
        return self.scalability_config.get("sql_validation", {}).get("enabled", False)
    
    @property
    def database_statistics_enabled(self) -> bool:
        """Check if database statistics collection is enabled"""
        return self.scalability_config.get("database_statistics", {}).get("enabled", False)
    
    # Redis Configuration
    @property
    def redis_url(self) -> str:
        """Get Redis URL for caching"""
        caching_config = self.scalability_config.get("caching", {})
        return caching_config.get("redis_url", "redis://localhost:6379/1")
    
    @property
    def celery_broker_url(self) -> str:
        """Get Celery broker URL"""
        async_config = self.scalability_config.get("async_processing", {})
        return async_config.get("celery_broker_url", "redis://localhost:6379/0")
    
    @property
    def celery_result_backend(self) -> str:
        """Get Celery result backend URL"""
        async_config = self.scalability_config.get("async_processing", {})
        return async_config.get("celery_result_backend", "redis://localhost:6379/0")
    
    # Connection Pool Configuration
    @property
    def connection_pool_config(self) -> Dict[str, Any]:
        """Get database connection pool configuration"""
        return self.scalability_config.get("connection_management", {
            "pool_size": 20,
            "pool_timeout": 30,
            "pool_recycle_hours": 1,
            "enable_pool_pre_ping": True
        })
    
    # LLM Configuration
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get("llm_settings", {})
    
    @property
    def primary_model(self) -> str:
        """Get primary LLM model"""
        return self.llm_config.get("primary_model", "gpt-4o")
    
    @property
    def cost_effective_model(self) -> str:
        """Get cost-effective model"""
        return self.llm_config.get("cost_effective_model", "gpt-4o-mini")
    
    # Agent Configuration
    @property
    def agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.get("agent", {})
    
    @property
    def max_iterations(self) -> int:
        """Get maximum agent iterations"""
        return self.agent_config.get("max_iterations", 10)
    
    # Cost Optimization Configuration
    @property
    def cost_optimization_config(self) -> Dict[str, Any]:
        """Get cost optimization configuration"""
        return self.config.get("cost_optimization", {})
    
    @property
    def cost_optimization_enabled(self) -> bool:
        """Check if cost optimization is enabled"""
        return self.cost_optimization_config.get("enable_optimization", True)
    
    # Environment Setup
    def setup_environment_variables(self):
        """Set up environment variables from configuration"""
        # Set database URL
        os.environ["DATABASE_URL"] = self.database_url
        
        # Set Redis URLs
        if self.caching_enabled:
            os.environ["REDIS_URL"] = self.redis_url
        
        if self.async_processing_enabled:
            os.environ["CELERY_BROKER_URL"] = self.celery_broker_url
            os.environ["CELERY_RESULT_BACKEND"] = self.celery_result_backend
        
        # Set feature flags
        os.environ["ASYNC_PROCESSING_ENABLED"] = str(self.async_processing_enabled)
        os.environ["CACHING_ENABLED"] = str(self.caching_enabled)
        os.environ["SQL_VALIDATION_ENABLED"] = str(self.sql_validation_enabled)
        os.environ["DATABASE_STATISTICS_ENABLED"] = str(self.database_statistics_enabled)
        
        logger.info("Environment variables configured from config.json")
    
    def get_startup_summary(self) -> str:
        """Get a summary of startup configuration"""
        summary = []
        summary.append("=== NLQ PLM System Configuration ===")
        summary.append(f"Database: {self.database_type.upper()} ({self.database_url})")
        summary.append(f"Primary Model: {self.primary_model}")
        summary.append(f"Cost-Effective Model: {self.cost_effective_model}")
        summary.append("")
        summary.append("=== Scalability Features ===")
        summary.append(f"Async Processing: {'✓' if self.async_processing_enabled else '✗'}")
        summary.append(f"Multi-Level Caching: {'✓' if self.caching_enabled else '✗'}")
        summary.append(f"SQL Validation: {'✓' if self.sql_validation_enabled else '✗'}")
        summary.append(f"Database Statistics: {'✓' if self.database_statistics_enabled else '✗'}")
        summary.append(f"Cost Optimization: {'✓' if self.cost_optimization_enabled else '✗'}")
        summary.append("")
        
        if self.caching_enabled:
            summary.append(f"Redis URL: {self.redis_url}")
        
        if self.async_processing_enabled:
            summary.append(f"Celery Broker: {self.celery_broker_url}")
            
        return "\n".join(summary)

# Global configuration instance
startup_config = None

def get_startup_config(config_path: Optional[str] = None) -> StartupConfig:
    """Get or create the global startup configuration instance"""
    global startup_config
    
    if startup_config is None:
        startup_config = StartupConfig(config_path)
        startup_config.setup_environment_variables()
    
    return startup_config

def initialize_system_config(config_path: Optional[str] = None):
    """Initialize the system configuration at startup"""
    config = get_startup_config(config_path)
    
    # Print startup summary
    print(config.get_startup_summary())
    
    return config
