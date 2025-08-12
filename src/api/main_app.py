
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import uvicorn
import time
import tempfile
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, inspect
from datetime import datetime
import logging

import yaml
import json

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import enhanced components
from src.core.agent.graph import build_agent_graph, initialize_enhanced_system, APP_STATE
from src.core.agent.agent_state import AgentState
from src.core.data_processing.utils import upload_files_to_db
from src.core.data_processing.vectors import HierarchicalRAGSystem
from src.core.agent.memory import ConversationVectorMemory
from src.core.agent.session_logger import get_session_logger, reset_session_logger
from src.core.agent.llm_tracker import get_global_tracker, reset_global_tracker
from src.core.agent.query_classifier import get_query_classifier, get_model_router
from src.config.startup_config import get_startup_config, initialize_system_config

# Import new scalability components
from src.core.caching.redis_cache import get_cache_system, reset_cache_system
from src.core.data_processing.tasks import TaskManager
try:
    from src.core.data_processing.tasks import process_uploaded_files
    CELERY_TASKS_AVAILABLE = True
except ImportError:
    process_uploaded_files = None
    CELERY_TASKS_AVAILABLE = False
from src.core.sql.sql_validator import get_sql_validator, reset_sql_validator
from src.core.sql.database_statistics import get_database_statistics, reset_database_statistics

# Load configuration with enhanced defaults
CONFIG_PATH = "config.json"
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print(f"Warning: Config file {CONFIG_PATH} not found, using intelligent defaults")
    CONFIG = {
        "api": {"host": "127.0.0.1", "port": 8000, "title": "Enhanced NLQ Intelligence System"},
        "openai": {"chat_model": "gpt-4o-mini", "temperature": 0},
        "database": {"default_url": "sqlite:///data/raw/plm_updated.db"},
        "memory": {"max_turns": 20, "enable_vector_search": True},
        "rag_system": {"storage_dir": "rag_storage", "use_cache": True},
        "query_classification": {"enable_cost_optimization": True},
        "logging": {"level": "INFO", "enhanced_tracking": True}
    }

# Enhanced logging configuration
log_config = CONFIG.get('logging', {})
logging.basicConfig(
    level=getattr(logging, log_config.get('level', 'INFO')),
    format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger = logging.getLogger(__name__)

# Global State for Enhanced System
AGENT_GRAPH = None
CONVERSATION_MEMORY = None
SESSION_LOGGER = None
RAG_SYSTEM = None
QUERY_CLASSIFIER = None
MODEL_ROUTER = None
STARTUP_CONFIG = None
ENGINE = None

# New scalability components
REDIS_CACHE = None
CACHE_LAYER = None
SQL_VALIDATOR = None
DB_STATISTICS = None
TASK_MANAGER = TaskManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management with intelligent system initialization."""
    global AGENT_GRAPH, CONVERSATION_MEMORY, SESSION_LOGGER, RAG_SYSTEM, QUERY_CLASSIFIER, MODEL_ROUTER
    global REDIS_CACHE, CACHE_LAYER, SQL_VALIDATOR, DB_STATISTICS, STARTUP_CONFIG, ENGINE
    
    try:
        logger.info("🚀 Initializing Enhanced NLQ Intelligence System with Scalability Features...")
        
        # Initialize startup configuration
        STARTUP_CONFIG = initialize_system_config()
        logger.info("✅ Startup configuration loaded")
        
        # Initialize Redis caching system
        try:
            REDIS_CACHE, CACHE_LAYER = get_cache_system(CONFIG)
            cache_health = REDIS_CACHE.health_check()
            logger.info(f"[PASS] Multi-level caching system initialized - Status: {cache_health['overall_status']}")
        except Exception as e:
            logger.warning(f"Cache system initialization failed: {e}, continuing without caching")
        
        # Initialize session logger with enhanced tracking
        SESSION_LOGGER = get_session_logger(CONFIG)
        logger.info(f"[PASS] Enhanced session logger initialized - Log file: {SESSION_LOGGER.log_file_path}")
        
        # Initialize enhanced conversation memory
        memory_config = CONFIG.get('memory', {})
        max_turns = memory_config.get('max_turns', 20)
        CONVERSATION_MEMORY = ConversationVectorMemory(max_turns=max_turns, config_path=CONFIG_PATH)
        logger.info(f"[PASS] Enhanced conversation memory initialized ({max_turns} turns, vector search enabled)")
        
        # Initialize query classifier and model router
        QUERY_CLASSIFIER = get_query_classifier()
        MODEL_ROUTER = get_model_router()
        logger.info("[PASS] Intelligent query classification and model routing systems initialized")
        
        # Build enhanced agent graph
        AGENT_GRAPH = build_agent_graph()
        logger.info("[PASS] Enhanced agent graph with intelligence routing built successfully")
        
        # Initialize database connection and stats
        db_config = CONFIG.get('database', {})
        DB_URL = os.getenv("DB_URL") or db_config.get('default_url', "sqlite:///data/raw/plm_updated.db")
        
        # Support PostgreSQL configuration
        if os.getenv("USE_POSTGRESQL", "").lower() == "true":
            postgresql_url = db_config.get('postgresql_url')
            if postgresql_url:
                DB_URL = postgresql_url
                logger.info("Using PostgreSQL database for enhanced performance")
        
        try:
            # Enhanced engine configuration for scalability
            engine_config = {
                'pool_size': db_config.get('connection_pool_size', 20),
                'pool_timeout': db_config.get('connection_timeout', 30),
                'pool_recycle': 3600,  # Recycle connections every hour
                'pool_pre_ping': True,  # Validate connections before use
                'echo': db_config.get('enable_query_logging', False)
            }
            
            if 'postgresql' in DB_URL.lower():
                engine_config['pool_size'] = 20  # Higher pool for PostgreSQL
                logger.info("PostgreSQL detected - using optimized connection pooling")
            elif 'sqlite' in DB_URL.lower():
                # SQLite-specific optimizations
                engine_config = {
                    'pool_timeout': 20,
                    'pool_recycle': -1,
                    'echo': db_config.get('enable_query_logging', False)
                }
                logger.info("SQLite detected - using compatible connection settings")
            
            ENGINE = create_engine(DB_URL, **engine_config)
            inspector = inspect(ENGINE)
            table_names = inspector.get_table_names()
            
            # Initialize SQL validator
            try:
                SQL_VALIDATOR = get_sql_validator(ENGINE, CONFIG)
                logger.info("[PASS] Advanced SQL validator initialized with schema validation")
            except Exception as e:
                logger.warning(f"SQL validator initialization failed: {e}")
            
            # Initialize database statistics collector
            try:
                if db_config.get('enable_statistics_collection', False):
                    DB_STATISTICS = get_database_statistics(ENGINE, CONFIG)
                    logger.info("[PASS] Database statistics collector initialized")
                else:
                    logger.info("[INFO] Database statistics collection disabled in config")
            except Exception as e:
                logger.warning(f"Database statistics initialization failed: {e}")
            
            # Initialize RAG system if tables exist
            if table_names:
                try:
                    RAG_SYSTEM = HierarchicalRAGSystem(ENGINE, CONFIG_PATH)
                    APP_STATE["rag_system"] = RAG_SYSTEM
                    logger.info(f"[PASS] Connected to database with {len(table_names)} tables")
                    logger.info("[PASS] Enhanced Hierarchical RAG System initialized")
                    
                    # ENHANCED: Try to initialize enhanced multiple retrieval system
                    try:
                        from src.core.integration.multi_retrieval_integration import create_enhanced_retrieval_orchestrator
                        enhanced_orchestrator = create_enhanced_retrieval_orchestrator(ENGINE)
                        APP_STATE["enhanced_orchestrator"] = enhanced_orchestrator
                        logger.info("[PASS] ✅ Enhanced Multiple Retrieval System initialized!")
                    except Exception as e:
                        logger.warning(f"[INFO] Enhanced retrieval system not available: {e}")
                    
                    # Also initialize the agent graph's APP_STATE
                    from src.core.agent.graph import initialize_enhanced_system
                    initialize_enhanced_system(ENGINE)
                    logger.info("[PASS] Agent graph enhanced system initialized")
                    
                except Exception as e:
                    logger.warning(f"RAG system initialization failed: {e}")
                    logger.info("[INFO] Continuing without RAG system - queries will use basic SQL generation")
                    APP_STATE["rag_system"] = None
                
                # Collect initial database statistics if enabled
                if DB_STATISTICS:
                    try:
                        stats_summary = DB_STATISTICS.collect_all_statistics()
                        logger.info(f"[PASS] Database statistics collected: {stats_summary['tables_analyzed']} tables analyzed")
                    except Exception as e:
                        logger.warning(f"Initial statistics collection failed: {e}")
            else:
                logger.info("[INFO] Database connected but no tables found - waiting for data upload")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise RuntimeError(f"Failed to connect to database: {e}")
        
        logger.info("✅ Enhanced NLQ Intelligence System with Scalability Features fully operational!")
        logger.info("🎯 Features enabled: Multi-level caching, async processing, SQL validation, database statistics")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced system: {e}")
        raise
    finally:
        # Enhanced cleanup
        if SESSION_LOGGER:
            SESSION_LOGGER.finalize_session()
        if REDIS_CACHE:
            try:
                cache_stats = REDIS_CACHE.get_cache_stats()
                if cache_stats and 'metrics' in cache_stats and 'hit_rate' in cache_stats['metrics']:
                    logger.info(f"Cache stats on shutdown - Hit rate: {cache_stats['metrics']['hit_rate']:.1f}%")
                else:
                    logger.info("Cache stats on shutdown - Stats not available")
            except Exception as e:
                logger.warning(f"Could not retrieve cache stats on shutdown: {e}")
        logger.info("[EXIT] Enhanced system shutting down gracefully")# Initialize FastAPI app with enhanced configuration
api_config = CONFIG.get('api', {})
app = FastAPI(
    title=api_config.get('title', "Enhanced NLQ Intelligence System"),
    description="Advanced Natural Language Query system with intelligent routing and cost optimization",
    version="2.0.0",
    lifespan=lifespan
)

# Add validation error handler for better debugging
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error on {request.method} {request.url}: {exc}")
    logger.error(f"Request body validation errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": str(exc.body) if hasattr(exc, 'body') else None,
            "message": "Request validation failed"
        }
    )

# Enhanced Pydantic Models
class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []  # Default to empty list
    options: Optional[Dict[str, Any]] = {}  # Enhanced options support
    page_number: Optional[int] = 1  # For pagination
    page_size: Optional[int] = 1000  # For pagination
    
    @field_validator('page_number', mode='before')
    @classmethod
    def validate_page_number(cls, v):
        # Handle invalid page_number values
        if isinstance(v, dict) or v is None:
            return 1
        try:
            return int(v)
        except (ValueError, TypeError):
            return 1
    
    @field_validator('page_size', mode='before') 
    @classmethod
    def validate_page_size(cls, v):
        # Handle invalid page_size values
        if isinstance(v, dict) or v is None:
            return 1000
        try:
            return int(v)
        except (ValueError, TypeError):
            return 1000
    
    class Config:
        # Allow extra fields and ignore them
        extra = "ignore"

class ClarificationRequest(BaseModel):
    session_id: str
    clarification_response: str
    selected_option: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    answer: str
    sql: Optional[str] = None
    graph: Optional[str] = None
    context_used: Optional[str] = None
    suggested_queries: Optional[List[Dict[str, str]]] = None
    # Enhanced response fields
    processing_info: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    cost_info: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    # HITL fields
    requires_clarification: Optional[bool] = None
    clarification_type: Optional[str] = None
    clarification_options: Optional[List[Dict[str, Any]]] = None
    # Pagination fields
    pagination_info: Optional[Dict[str, Any]] = None
    # SQL Optimization fields
    sql_optimization: Optional[Dict[str, Any]] = None
    optimization_recommendations: Optional[List[Dict[str, str]]] = None
    original_sql: Optional[str] = None
    optimization_applied: Optional[bool] = None

class SystemStatus(BaseModel):
    status: str
    components: Dict[str, bool]
    performance: Dict[str, Any]
    version: str = "2.0.0"

# Enhanced API Endpoints

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serves the enhanced chat interface."""
    return FileResponse("src/ui/index_improved.html")

@app.post("/upload", status_code=202)
async def enhanced_async_upload_data_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(None),
    context_file: Optional[UploadFile] = File(None),
    options: Optional[Dict[str, Any]] = None
):
    """Enhanced asynchronous file upload with scalable background processing."""
    global RAG_SYSTEM, TASK_MANAGER
    
    # Validate input
    if not files and not context_file:
        # Check if we have existing data
        db_config = CONFIG.get('database', {})
        DB_URL = os.getenv("DB_URL") or db_config.get('default_url', "sqlite:///data/raw/plm_updated.db")
        
        try:
            engine = create_engine(DB_URL)
            inspector = inspect(engine)
            existing_tables = inspector.get_table_names()
            
            if existing_tables:
                return {
                    "message": f"System ready with existing {len(existing_tables)} tables",
                    "status": "ready",
                    "processing_mode": "existing_data",
                    "tables": existing_tables[:10]  # Show first 10 tables
                }
            else:
                raise HTTPException(status_code=400, detail="No files provided and no existing data found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database validation failed: {str(e)}")
    
    try:
        # Create temporary directory for file storage
        temp_dir = tempfile.mkdtemp(prefix="nlq_upload_")
        temp_files = []
        context_file_path = None
        
        # Save uploaded files to temporary location
        if files:
            for file in files:
                if not file.filename:
                    continue
                    
                temp_file_path = os.path.join(temp_dir, file.filename)
                
                with open(temp_file_path, 'wb') as temp_file:
                    shutil.copyfileobj(file.file, temp_file)
                
                temp_files.append(temp_file_path)
                logger.info(f"Saved {file.filename} to temporary storage")
        
        # Save context file if provided
        if context_file and context_file.filename:
            context_file_path = os.path.join(temp_dir, context_file.filename)
            
            with open(context_file_path, 'wb') as temp_file:
                shutil.copyfileobj(context_file.file, temp_file)
            
            logger.info(f"Saved context file {context_file.filename}")
        
        # Get database URL for task
        db_config = CONFIG.get('database', {})
        DB_URL = os.getenv("DB_URL") or db_config.get('default_url', "sqlite:///data/raw/plm_updated.db")
        
        # Start asynchronous processing task
        try:
            task_options = options or {}
            task_options['config_path'] = CONFIG_PATH
            
            # Disable Celery on Windows due to billiard issues - use FastAPI background tasks only
            # Submit task to Celery (if available) or use background tasks
            # task_result = process_uploaded_files.delay(
            #     file_paths=temp_files,
            #     context_file_path=context_file_path,
            #     db_url=DB_URL,
            #     user_id=None,  # Could add user tracking
            #     options=task_options
            # )
            # 
            # task_id = task_result.id
            # processing_mode = "async_celery"
            
            # Force use of FastAPI background task to avoid Windows Celery issues
            raise Exception("Using FastAPI background task instead of Celery")
            
        except Exception as e:
            logger.warning(f"Using FastAPI background task for processing (Celery disabled for Windows)")
            
            # Fallback to FastAPI background task
            task_id = f"bg_{int(time.time())}"
            
            async def fallback_processing():
                try:
                    from src.core.data_processing.utils import upload_files_to_db
                    from src.core.data_processing.vectors import HierarchicalRAGSystem, build_scalable_retriever_system
                    
                    engine = create_engine(DB_URL)
                    
                    # Only process data files if they exist
                    if temp_files:
                        # Process files
                        tables = upload_files_to_db(temp_files, engine)
                        logger.info(f"Processed {len(temp_files)} data files, created {len(tables)} tables")
                    else:
                        # No data files, just get existing tables
                        inspector = inspect(engine)
                        tables = inspector.get_table_names()
                        logger.info(f"Schema-only upload: using {len(tables)} existing tables")
                    
                    # Process context file if available
                    if context_file_path:
                        try:
                            with open(context_file_path, 'r', encoding='utf-8') as f:
                                if context_file_path.endswith('.json'):
                                    context_data = json.load(f)
                                else:
                                    context_data = {"description": f.read()}
                            
                            # Build/rebuild RAG system with context
                            logger.info("Initializing HierarchicalRAGSystem...")
                            rag_system = HierarchicalRAGSystem(engine, CONFIG_PATH)
                            logger.info("HierarchicalRAGSystem initialized successfully")
                            
                            if context_data and tables:
                                logger.info("Building scalable retriever system...")
                                try:
                                    retrievers = build_scalable_retriever_system(engine, context_data)
                                    logger.info(f"Enhanced RAG system with retrievers: {type(retrievers)}")
                                    if isinstance(retrievers, tuple):
                                        logger.info(f"Retriever tuple length: {len(retrievers)}")
                                        logger.info(f"Retriever types: {[type(r) for r in retrievers]}")
                                except Exception as ret_error:
                                    logger.error(f"Retriever building failed: {ret_error}")
                                    raise
                            else:
                                logger.info("Basic RAG system initialized")
                                
                        except Exception as ctx_error:
                            logger.error(f"Context processing failed: {ctx_error}")
                            logger.exception("Full context processing error:")
                            context_data = None
                    else:
                        context_data = None
                        # No context file, but we still need to ensure RAG system exists
                        logger.info("No context file, initializing basic RAG system...")
                        rag_system = HierarchicalRAGSystem(engine, CONFIG_PATH)
                        logger.info("RAG system initialized with existing database tables")
                    
                    # Update both global RAG systems and agent graph APP_STATE
                    global RAG_SYSTEM
                    if 'rag_system' in locals():
                        RAG_SYSTEM = rag_system
                        APP_STATE["rag_system"] = rag_system
                        logger.info("Updated global RAG system and APP_STATE")
                        
                        # CRITICAL: Update the agent graph's APP_STATE as well
                        from src.core.agent.graph import initialize_enhanced_system
                        initialize_enhanced_system(engine, context_data)
                        logger.info("Updated agent graph APP_STATE with enhanced system")
                    
                    # Cleanup temp files
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                    logger.info(f"Background processing completed for task {task_id}")
                    
                except Exception as e:
                    logger.error(f"Background processing failed for task {task_id}: {e}")
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            background_tasks.add_task(fallback_processing)
            processing_mode = "background_task"
        
        # Return immediate response
        response = {
            "message": "Files received and processing started",
            "status": "processing",
            "task_id": task_id,
            "processing_mode": processing_mode,
            "files_received": len(temp_files),
            "context_file_provided": context_file_path is not None,
            "estimated_processing_time": f"{len(temp_files) * 30}-{len(temp_files) * 60} seconds",
            "check_status_endpoint": f"/task/status/{task_id}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache task info for status checking
        if CACHE_LAYER:
            CACHE_LAYER.cache.set(
                'task_status',
                task_id,
                {
                    'status': 'processing',
                    'created_at': datetime.now().isoformat(),
                    'files_count': len(temp_files)
                },
                ttl=3600  # 1 hour
            )
        
        logger.info(f"Started async processing task {task_id} for {len(temp_files)} files")
        return response
        
    except Exception as e:
        # Cleanup on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.error(f"Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@app.get("/task/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of an asynchronous task."""
    try:
        # Check task status using TaskManager
        status_info = TASK_MANAGER.get_task_status(task_id)
        
        # Enhance with cached information if available
        if CACHE_LAYER:
            cached_info = CACHE_LAYER.cache.get('task_status', task_id)
            if cached_info:
                status_info.update(cached_info)
        
        return {
            "task_id": task_id,
            "status_info": status_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Task status check failed for {task_id}: {e}")
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task."""
    try:
        result = TASK_MANAGER.cancel_task(task_id)
        return {
            "task_id": task_id,
            "cancellation_result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Task cancellation failed for {task_id}: {e}")
        return {
            "task_id": task_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/tasks/active")
async def get_active_tasks():
    """Get list of active processing tasks."""
    try:
        active_tasks = TASK_MANAGER.get_active_tasks()
        return {
            "active_tasks": active_tasks,
            "count": len(active_tasks),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        # Initialize enhanced system state
        enhanced_state = initialize_enhanced_system(ENGINE, schema_description_data)
        
        logger.info("[PASS] Enhanced RAG system built with intelligent retrievers")
        
        # Clear and reset memory for new session
        if CONVERSATION_MEMORY:
            CONVERSATION_MEMORY.history.clear()
            CONVERSATION_MEMORY.embeddings.clear()
            logger.info("[PASS] Conversation memory cleared for new session")
        
        # Reset session tracking
        global SESSION_LOGGER
        SESSION_LOGGER = reset_session_logger(CONFIG)
        reset_global_tracker()
        logger.info(f"[PASS] Enhanced session logging initialized - Log file: {SESSION_LOGGER.log_file_path}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Enhanced response with system information
        response_data = {
            "message": f"✅ Enhanced system ready! Processed {len(files) if files else 0} files in {processing_time:.2f}s",
            "system_info": {
                "rag_layers": len([r for r in retrievers.values() if r is not None]),
                "tables_available": len(inspector.get_table_names()),
                "intelligence_features": ["query_classification", "model_routing", "cost_optimization", "semantic_search"],
                "processing_time": f"{processing_time:.2f}s"
            }
        }
        
        if context_file:
            response_data["context_loaded"] = True
            
        return response_data
        
    except Exception as e:
        logger.error(f"Enhanced upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced processing failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat_endpoint(payload: ChatRequest):
    """Enhanced chat endpoint with intelligent processing, caching, and optimization."""
    logger.info(f"Chat endpoint called with query: {payload.query}")
    logger.info(f"AGENT_GRAPH is None: {AGENT_GRAPH is None}")
    logger.info(f"CONVERSATION_MEMORY is None: {CONVERSATION_MEMORY is None}")
    logger.info(f"SESSION_LOGGER is None: {SESSION_LOGGER is None}")
    
    if AGENT_GRAPH is None or CONVERSATION_MEMORY is None or SESSION_LOGGER is None:
        logger.error("Enhanced system components not available")
        raise HTTPException(status_code=503, detail="Enhanced system components not available.")
    
    # Debug logging to understand current state
    rag_system_status = APP_STATE.get("rag_system")
    logger.info(f"Chat endpoint called - RAG system status: {rag_system_status is not None}")
    logger.info(f"Global RAG_SYSTEM status: {RAG_SYSTEM is not None}")
    
    # Use either APP_STATE or global RAG_SYSTEM
    current_rag_system = APP_STATE.get("rag_system") or RAG_SYSTEM
    
    if current_rag_system is None:
        logger.warning("No RAG system available, returning default message")
        return ChatResponse(
            answer="🤖 Enhanced NLQ system is ready! Please upload your data file to begin intelligent analysis.",
            processing_info={
                "status": "waiting_for_data", 
                "system_version": "2.0.0",
                "debug_info": f"APP_STATE rag_system: {rag_system_status}, Global RAG_SYSTEM: {RAG_SYSTEM is not None}"
            }
        )

    start_time = datetime.now()
    processing_info = {"start_time": start_time.isoformat()}
    
    # Multi-level caching strategy
    cache_key = None
    cached_response = None
    
    if CACHE_LAYER:
        try:
            # Generate cache key from query and context
            import hashlib
            query_context = {
                'query': payload.query.lower().strip(),
                'history_hash': hashlib.md5(str(payload.history).encode()).hexdigest()[:8],
                'options': payload.options
            }
            cache_key = hashlib.md5(str(query_context).encode()).hexdigest()
            
            # Check for cached final response
            cached_response = CACHE_LAYER.get_final_response(cache_key)
            
            if cached_response:
                logger.info(f"Cache hit for query: {payload.query[:50]}...")
                
                # Update with cached response info
                cached_response['processing_info']['cached'] = True
                cached_response['processing_info']['cache_hit_time'] = datetime.now().isoformat()
                
                return ChatResponse(**cached_response)
                
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
    
    try:
        # Reset tracking for this query
        get_global_tracker().reset_all()
        
        # Get enhanced memory configuration
        memory_config = CONFIG.get('memory', {})
        use_vector_search = memory_config.get('enable_vector_search', True)
        
        # Retrieve enhanced context with intelligence
        vector_contexts = CONVERSATION_MEMORY.search(
            payload.query, 
            top_k=5,  # Increased for better context
            use_retrieval=use_vector_search
        )
        
        # Build enhanced context summary
        context_summary = ""
        if vector_contexts:
            context_summary = f"Relevant context from {len(vector_contexts)} previous conversations:\n"
            context_summary += "\n".join([
                f"- Q: {c['query'][:80]}... A: {c['response'][:80]}..." 
                for c in vector_contexts[:3]
            ])

        # Create enhanced initial state
        initial_state = AgentState(
            user_query=payload.query,
            chat_history=payload.history,
            context_history=vector_contexts

        )
        
        # Add pagination info if provided
        processing_info["pagination"] = {
            "page_number": payload.page_number,
            "page_size": payload.page_size,
            "pagination_requested": payload.page_number > 1 or payload.page_size < 1000
        }
        
        # Add performance tracking
        processing_info["memory_contexts"] = len(vector_contexts)
        processing_info["vector_search_enabled"] = use_vector_search
        processing_info["caching_enabled"] = CACHE_LAYER is not None
        processing_info["sql_validation_enabled"] = SQL_VALIDATOR is not None
        
        # Invoke enhanced agent graph with validation
        logger.info(f"Invoking agent graph with query: {payload.query}")
        logger.info(f"Initial state type: {type(initial_state)}")
        try:
            final_state = AGENT_GRAPH.invoke(initial_state)
            logger.info(f"Agent graph invocation completed successfully")
            logger.info(f"Final state type: {type(final_state)}")
        except Exception as graph_error:
            logger.error(f"Agent graph invocation failed: {graph_error}")
            logger.exception("Full agent graph error:")
            raise
        
        # Extract enhanced results
        plain_text_answer = final_state.get('final_answer', "I encountered an issue processing your request.")
        sql_query = final_state.get('sql_query')
        sql_result = final_state.get('sql_result')
        graph_data = final_state.get('graph_b64')
        
        # Advanced SQL validation if available
        sql_validation_report = None
        if sql_query and SQL_VALIDATOR:
            try:
                sql_validation_report = SQL_VALIDATOR.validate(sql_query)
                
                if not sql_validation_report.is_valid:
                    logger.warning(f"SQL validation failed: {sql_validation_report.errors}")
                    # Could optionally retry with corrected SQL
                
                processing_info["sql_validation"] = {
                    "is_valid": sql_validation_report.is_valid,
                    "performance_score": sql_validation_report.performance_score,
                    "warnings": sql_validation_report.warnings,
                    "complexity": sql_validation_report.query_complexity
                }
                
            except Exception as e:
                logger.warning(f"SQL validation failed: {e}")
        
        # Cache SQL result if successful
        if (sql_query and sql_result is not None and 
            not (hasattr(sql_result, 'empty') and sql_result.empty) and CACHE_LAYER):
            try:
                sql_hash = hashlib.md5(sql_query.encode()).hexdigest()
                execution_time = final_state.get('sql_execution_time', 0)
                CACHE_LAYER.cache_sql_result(sql_hash, sql_result, execution_time)
            except Exception as e:
                logger.debug(f"SQL result caching failed: {e}")
        
        # Get processing summary
        processing_summary = final_state.get_processing_summary() if hasattr(final_state, 'get_processing_summary') else {}
        processing_info.update(processing_summary)
        
        # Format response for UI with enhanced features
        from src.core.agent.sql_logic import format_response_for_ui
        ui_formatted_answer = format_response_for_ui(plain_text_answer, sql_result, payload.query)
        
        # Get comprehensive LLM usage statistics
        llm_calls = get_global_tracker().get_all_calls()
        llm_stats = get_global_tracker().get_session_stats()
        
        # Calculate performance metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        performance_metrics = {
            "processing_time_seconds": processing_time,
            "total_llm_calls": len(llm_calls),
            "total_tokens": llm_stats.get("total_tokens", 0),
            "sql_retries": final_state.get('retry_count', 0),
            "context_retrieval_time": processing_info.get("context_retrieval_time", 0),
            "query_complexity": final_state.query_classification.complexity.value if hasattr(final_state, 'query_classification') and final_state.query_classification else "unknown"
        }
        
        # Calculate cost information
        cost_info = {
            "estimated_cost_usd": final_state.get('total_cost_estimate', 0),
            "token_breakdown": llm_stats,
            "cost_per_token": 0.000015 if processing_summary.get("model_used") == "gpt-4o" else 0.0000015
        }
        
        # Enhanced session logging
        SESSION_LOGGER.log_query_session(
            user_query=payload.query,
            sql_query=sql_query,
            agent_response=plain_text_answer,
            llm_calls=llm_calls,
            context_used=context_summary,
            error=None,
            metadata={
                "enhanced_features": True,
                "system_version": "2.0.0",
                "performance_metrics": performance_metrics,
                "cost_info": cost_info,
                "query_classification": processing_summary.get("query_complexity", "unknown"),
                "model_routing": processing_summary.get("model_used", "unknown"),
                "graph_generated": graph_data is not None,
                "intelligence_features_used": [
                    "query_classification",
                    "model_routing", 
                    "enhanced_rag",
                    "smart_context_retrieval"
                ]
            }
        )

        # Store in enhanced conversation memory
        CONVERSATION_MEMORY.add(
            user_query=payload.query,
            agent_response=plain_text_answer,
            sql=sql_query
        )
        
        # Prepare final response with enhanced features
        pagination_info = None
        optimization_info = None
        optimization_recommendations = None
        original_sql = None
        optimization_applied = False
        
        if isinstance(sql_result, dict) and 'pagination_info' in str(sql_result):
            # Handle paginated results
            pagination_info = {
                "page": payload.page_number,
                "page_size": payload.page_size,
                "total_rows": sql_result.get('total_rows'),
                "has_more": sql_result.get('has_more', False),
                "total_pages": sql_result.get('total_pages')
            }
            
            # Extract optimization info if available
            if 'optimization_analysis' in sql_result:
                optimization_info = sql_result['optimization_analysis']
                optimization_recommendations = sql_result.get('optimization_recommendations', [])
                original_sql = sql_result.get('original_sql')
                optimization_applied = sql_result.get('optimization_applied', False)
        
        # Also check final state for optimization data
        if not optimization_info and hasattr(final_state, 'optimization_analysis') and final_state.optimization_analysis:
            optimization_info = final_state.optimization_analysis
            optimization_recommendations = final_state.optimization_recommendations or []
            original_sql = final_state.original_sql_query
            optimization_applied = final_state.sql_optimization_applied

        final_response = ChatResponse(
            answer=ui_formatted_answer,
            sql=sql_query,
            graph=graph_data,
            context_used=context_summary if context_summary else None,
            processing_info=processing_info,
            performance_metrics=performance_metrics,
            cost_info=cost_info,
            confidence_score=final_state.get('confidence_score', 0.8),
            # HITL fields
            requires_clarification=final_state.get('requires_clarification', False),
            clarification_type=final_state.get('clarification_type'),
            clarification_options=final_state.get('clarification_options', []),
            # Pagination fields
            pagination_info=pagination_info,
            # SQL Optimization fields
            sql_optimization=optimization_info,
            optimization_recommendations=optimization_recommendations,
            original_sql=original_sql,
            optimization_applied=optimization_applied
        )
        
        # Cache successful response for future use
        if cache_key and CACHE_LAYER:
            try:
                response_data = {
                    "answer": ui_formatted_answer,
                    "sql": sql_query,
                    "graph": graph_data,
                    "context_used": context_summary,
                    "processing_info": processing_info,
                    "performance_metrics": performance_metrics,
                    "cost_info": cost_info,
                    "confidence_score": final_state.get('confidence_score', 0.8)
                }
                
                CACHE_LAYER.cache_final_response(cache_key, response_data)
                logger.debug(f"Cached successful response for query: {payload.query[:50]}...")
                
            except Exception as e:
                logger.debug(f"Response caching failed: {e}")
        
        return final_response
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        error_occurred = str(e)
        logger.error(f"Enhanced chat processing failed: {e}")
        
        # Enhanced error response
        error_response = f"I encountered an error while processing your request. The enhanced system is analyzing the issue. Please try rephrasing your question. Error: {str(e)}"
        
        # Log enhanced error session
        llm_calls = get_global_tracker().get_all_calls()
        processing_time = (datetime.now() - start_time).total_seconds()
        
        SESSION_LOGGER.log_query_session(
            user_query=payload.query,
            sql_query=None,
            agent_response=error_response,
            llm_calls=llm_calls,
            context_used=None,
            error=error_occurred,
            metadata={
                "enhanced_system": True,
                "error_type": type(e).__name__,
                "processing_time_seconds": processing_time,
                "system_version": "2.0.0"
            }
        )
        
        CONVERSATION_MEMORY.add(user_query=payload.query, agent_response=error_response)
        raise HTTPException(status_code=500, detail="Enhanced system encountered an processing error.")

@app.post("/chat/clarify", response_model=ChatResponse)
async def handle_clarification_response(payload: ClarificationRequest):
    """Handle user responses to clarification requests (HITL)."""
    if AGENT_GRAPH is None or CONVERSATION_MEMORY is None:
        raise HTTPException(status_code=503, detail="Enhanced system components not available.")
    
    try:
        # Reconstruct the query with clarification
        clarified_query = f"{payload.clarification_response}"
        if payload.selected_option:
            clarified_query += f" (Selected option: {payload.selected_option})"
        
        # Create new request with clarified query
        chat_request = ChatRequest(
            query=clarified_query,
            history=[],  # Fresh start after clarification
            options=payload.additional_context or {}
        )
        
        # Process the clarified request through normal chat flow
        return await enhanced_chat_endpoint(chat_request)
        
    except Exception as e:
        logger.error(f"Clarification handling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process clarification: {str(e)}")

@app.get("/system/status", response_model=SystemStatus)
async def get_enhanced_system_status():
    """Get comprehensive system status with intelligence metrics."""
    session_stats = SESSION_LOGGER.get_session_stats() if SESSION_LOGGER else {}
    llm_stats = get_global_tracker().get_session_stats()
    
    # Component status
    components = {
        "agent_graph": AGENT_GRAPH is not None,
        "conversation_memory": CONVERSATION_MEMORY is not None,
        "session_logger": SESSION_LOGGER is not None,
        "rag_system": APP_STATE.get("rag_system") is not None,
        "query_classifier": QUERY_CLASSIFIER is not None,
        "model_router": MODEL_ROUTER is not None,
        "database": APP_STATE.get("rag_system") is not None
    }
    
    # Performance metrics
    performance = {
        "session_stats": session_stats,
        "llm_usage": llm_stats,
        "system_uptime": datetime.now().isoformat(),
        "intelligence_features": [
            "query_classification",
            "model_routing",
            "cost_optimization", 
            "enhanced_rag",
            "semantic_memory",
            "performance_tracking"
        ],
        "active_components": sum(components.values()),
        "total_components": len(components)
    }
    
    # Overall system status
    status = "healthy" if all(components.values()) else "partial"
    
    return SystemStatus(
        status=status,
        components=components,
        performance=performance
    )

@app.get("/intelligence/metrics")
async def get_intelligence_metrics():
    """Get detailed intelligence and optimization metrics."""
    if not all([QUERY_CLASSIFIER, MODEL_ROUTER, RAG_SYSTEM]):
        raise HTTPException(status_code=503, detail="Intelligence systems not fully initialized")
    
    # Get comprehensive metrics
    metrics = {
        "query_classification": {
            "available": True,
            "supported_complexity_levels": ["simple", "moderate", "complex"],
            "supported_intents": ["lookup", "aggregation", "comparison", "analysis", "trend", "relationship"],
            "supported_domains": ["general", "financial", "sales", "inventory", "customer", "operational"]
        },
        "model_routing": {
            "available_models": ["gpt-4o-mini", "gpt-4o"],
            "routing_strategy": "cost_and_complexity_based",
            "cost_optimization": True
        },
        "rag_system": {
            "type": "hierarchical",
            "layers": ["business_rules", "schema", "samples", "statistics", "relationships", "query_patterns"],
            "caching_enabled": CONFIG.get('rag_system', {}).get('use_cache', True),
            "adaptive_retrieval": True
        },
        "memory_system": {
            "max_turns": CONFIG.get('memory', {}).get('max_turns', 20),
            "vector_search": CONFIG.get('memory', {}).get('enable_vector_search', True),
            "semantic_similarity": True
        },
        "performance_tracking": {
            "token_usage": True,
            "cost_estimation": True,
            "response_time": True,
            "quality_scoring": True
        }
    }
    
    return {
        "system_version": "2.0.0",
        "intelligence_capabilities": metrics,
        "status": "fully_operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def enhanced_health_check():
    """Enhanced health check with detailed system information."""
    health_info = {
        "status": "healthy",
        "system_version": "2.0.0",
        "components": {
            "agent_ready": AGENT_GRAPH is not None,
            "memory_ready": CONVERSATION_MEMORY is not None,
            "logger_ready": SESSION_LOGGER is not None,
            "rag_ready": APP_STATE.get("rag_system") is not None,
            "classifier_ready": QUERY_CLASSIFIER is not None,
            "router_ready": MODEL_ROUTER is not None
        },
        "intelligence_features": {
            "query_classification": True,
            "model_routing": True,
            "cost_optimization": True,
            "enhanced_rag": True,
            "semantic_memory": True,
            "performance_tracking": True
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Add session information if available
    if SESSION_LOGGER:
        health_info["session_info"] = SESSION_LOGGER.get_session_stats()
    
    # Check if all critical components are ready
    critical_components = ["agent_ready", "memory_ready", "logger_ready"]
    if not all(health_info["components"][comp] for comp in critical_components):
        health_info["status"] = "degraded"
    
    # Add scalability component status
    health_info["scalability_components"] = {
        "cache_system": CACHE_LAYER is not None,
        "sql_validator": SQL_VALIDATOR is not None,
        "database_statistics": DB_STATISTICS is not None,
        "task_manager": TASK_MANAGER is not None
    }
    
    return health_info

# Enhanced scalability endpoints

@app.get("/system/cache/stats")
async def get_cache_statistics():
    """Get detailed caching system statistics."""
    if not CACHE_LAYER:
        raise HTTPException(status_code=503, detail="Caching system not available")
    
    try:
        cache_stats = CACHE_LAYER.cache.get_cache_stats()
        cache_health = CACHE_LAYER.cache.health_check()
        
        return {
            "cache_statistics": cache_stats,
            "cache_health": cache_health,
            "cache_layers": {
                "sql_results": "Cached SQL query results",
                "llm_responses": "Cached LLM responses",
                "final_responses": "Cached complete responses",
                "rag_contexts": "Cached RAG contexts"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")

@app.post("/system/cache/clear")
async def clear_cache_system():
    """Clear all caches (admin operation)."""
    cleared_caches = []
    errors = []
    
    try:
        # Clear Redis cache if available
        if CACHE_LAYER:
            success = CACHE_LAYER.cache.clear_all()
            if success:
                cleared_caches.append("Redis cache")
            else:
                errors.append("Redis cache clearing failed")
        
        # Clear conversation memory
        if CONVERSATION_MEMORY:
            try:
                CONVERSATION_MEMORY.clear()
                cleared_caches.append("Conversation memory")
            except Exception as e:
                errors.append(f"Conversation memory: {str(e)}")
        
        # Clear RAG system cache
        global RAG_SYSTEM
        if RAG_SYSTEM and hasattr(RAG_SYSTEM, 'context_cache'):
            try:
                RAG_SYSTEM.context_cache.clear()
                cleared_caches.append("RAG context cache")
            except Exception as e:
                errors.append(f"RAG context cache: {str(e)}")
        
        # Reset LLM token tracker
        try:
            reset_global_tracker()
            cleared_caches.append("LLM token tracker")
        except Exception as e:
            errors.append(f"LLM token tracker: {str(e)}")
        
        return {
            "success": len(errors) == 0,
            "cleared_caches": cleared_caches,
            "errors": errors,
            "message": f"Cache clearing completed. Cleared: {len(cleared_caches)}, Errors: {len(errors)}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")

@app.post("/system/cache/clear-conversation")
async def clear_conversation_cache():
    """Clear only conversation-related caches."""
    try:
        cleared = []
        
        # Clear conversation memory
        if CONVERSATION_MEMORY:
            CONVERSATION_MEMORY.clear()
            cleared.append("Conversation memory")
        
        # Clear RAG context cache
        global RAG_SYSTEM
        if RAG_SYSTEM and hasattr(RAG_SYSTEM, 'context_cache'):
            RAG_SYSTEM.context_cache.clear()
            cleared.append("RAG context cache")
        
        return {
            "success": True,
            "cleared": cleared,
            "message": "Conversation caches cleared - your next query will generate fresh responses",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation cache: {str(e)}")

@app.get("/system/database/statistics")
async def get_database_statistics():
    """Get comprehensive database statistics and optimization insights."""
    if not DB_STATISTICS:
        raise HTTPException(status_code=503, detail="Database statistics collection not enabled")
    
    try:
        # Force refresh of statistics
        stats_summary = DB_STATISTICS.collect_all_statistics(force_refresh=True)
        optimization_insights = DB_STATISTICS.get_optimization_insights()
        
        return {
            "database_statistics": stats_summary,
            "optimization_insights": optimization_insights,
            "collection_info": {
                "collector_dialect": DB_STATISTICS.db_dialect,
                "cache_ttl_hours": DB_STATISTICS.cache_ttl.total_seconds() / 3600,
                "features_enabled": {
                    "row_counts": DB_STATISTICS.enable_row_counts,
                    "column_statistics": DB_STATISTICS.enable_column_stats,
                    "index_analysis": DB_STATISTICS.enable_index_analysis
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database statistics: {str(e)}")

@app.get("/system/sql/validation-summary")
async def get_sql_validation_summary():
    """Get SQL validation capabilities and summary."""
    if not SQL_VALIDATOR:
        raise HTTPException(status_code=503, detail="SQL validation not available")
    
    try:
        validation_summary = SQL_VALIDATOR.get_validation_summary()
        
        return {
            "sql_validation_summary": validation_summary,
            "validation_features": {
                "syntax_validation": "Comprehensive SQL syntax checking",
                "schema_validation": "Table and column existence verification",
                "security_validation": "SQL injection and forbidden operation detection",
                "performance_analysis": "Query performance scoring and optimization suggestions",
                "automatic_optimization": "Query optimization with suggestions"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get validation summary: {str(e)}")

class SQLOptimizationRequest(BaseModel):
    sql_query: str
    optimization_level: Optional[str] = "BASIC"  # BASIC, INTERMEDIATE, AGGRESSIVE
    explain_only: Optional[bool] = False

@app.post("/sql/optimize")
async def analyze_sql_optimization(request: SQLOptimizationRequest):
    """Analyze and optimize SQL queries with EXPLAIN plan analysis."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Database engine not available")
    
    try:
        # Import SQL optimizer components
        from src.core.agent.sql_logic import get_sql_optimizer, analyze_sql_optimization
        
        # Get optimizer instance
        optimizer = get_sql_optimizer()
        if not optimizer:
            raise HTTPException(status_code=503, detail="SQL optimizer not available")
        
        # Perform optimization analysis
        optimization_result = analyze_sql_optimization(
            sql_query=request.sql_query,
            engine=ENGINE,
            optimization_level=request.optimization_level,
            explain_only=request.explain_only
        )
        
        return {
            "status": "success",
            "original_query": request.sql_query,
            "optimization_analysis": optimization_result.get('analysis'),
            "optimization_recommendations": optimization_result.get('recommendations', []),
            "optimized_query": optimization_result.get('optimized_query') if not request.explain_only else None,
            "performance_improvement": optimization_result.get('performance_improvement'),
            "confidence_score": optimization_result.get('confidence_score'),
            "explain_plan": optimization_result.get('explain_plan'),
            "optimization_level": request.optimization_level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"SQL optimization analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization analysis failed: {str(e)}")

@app.get("/sql/optimizer/status")
async def get_sql_optimizer_status():
    """Get SQL optimizer status and capabilities."""
    try:
        from src.core.agent.sql_logic import get_sql_optimizer, SQL_OPTIMIZER_AVAILABLE
        
        if not SQL_OPTIMIZER_AVAILABLE:
            return {
                "status": "unavailable",
                "reason": "SQL optimizer not initialized or dependencies missing"
            }
        
        optimizer = get_sql_optimizer()
        if not optimizer:
            return {
                "status": "unavailable", 
                "reason": "Optimizer instance not available"
            }
        
        return {
            "status": "available",
            "optimizer_features": {
                "explain_plan_analysis": "Full EXPLAIN plan parsing and analysis",
                "query_rewriting": "Intelligent query structure optimization",
                "index_recommendations": "Smart index usage suggestions",
                "join_optimization": "Join order and type optimization",
                "predicate_optimization": "WHERE clause and filtering optimization",
                "performance_estimation": "Cost-based optimization with confidence scoring"
            },
            "supported_databases": ["SQLite", "PostgreSQL", "MySQL"],
            "optimization_levels": ["BASIC", "INTERMEDIATE", "AGGRESSIVE"],
            "safety_features": {
                "confidence_threshold": "Only applies optimizations with high confidence",
                "read_only_validation": "Ensures optimized queries don't modify data",
                "rollback_support": "Always preserves original query"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimizer status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    # Enhanced startup configuration
    api_config = CONFIG.get('api', {})
    host = api_config.get('host', '127.0.0.1')
    port = api_config.get('port', 8000)
    
    print("🚀 Starting Enhanced NLQ Intelligence System with Scalability Features...")
    print("=" * 70)
    print("🤖 Core Intelligence: Query Classification, Model Routing, Cost Optimization")
    print("🧠 RAG System: Hierarchical with 6 intelligent layers")
    print("💭 Memory: Enhanced semantic search with 20-turn capacity")
    print("📊 Performance: Comprehensive tracking and cost monitoring")
    print("🚀 Scalability: Multi-level caching, async processing, SQL validation")
    print("💾 Database: Advanced statistics collection and optimization")
    print("⚡ Processing: Asynchronous task queue with Redis backend")
    print("=" * 70)
    print(f"🌐 Enhanced UI: http://{host}:{port}")
    print(f"📈 Intelligence Metrics: http://{host}:{port}/intelligence/metrics")
    print(f"📊 Cache Statistics: http://{host}:{port}/system/cache/stats")
    print(f"🗄️  Database Stats: http://{host}:{port}/system/database/statistics")
    print(f"✅ SQL Validation: http://{host}:{port}/system/sql/validation-summary")
    print("=" * 70)
    
    uvicorn.run(app, host=host, port=port, log_level="info")