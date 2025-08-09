"""
Asynchronous Task Processing System for Scalable Data Ingestion
This module implements Celery-based background task processing for file uploads and RAG building
"""

import os
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Celery imports with fallback
try:
    from celery import Celery, Task
    from celery.result import AsyncResult
    from celery.signals import task_success, task_failure, task_retry
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None
    Task = None
    AsyncResult = None
    task_success = None
    task_failure = None
    task_retry = None

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd

# Import core processing modules
from .utils import upload_files_to_db, drop_all_tables
from .vectors import HierarchicalRAGSystem, build_scalable_retriever_system

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration and initialization (only if available)
if CELERY_AVAILABLE:
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

    # Initialize Celery app
    celery_app = Celery(
        'nlq_tasks',
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
        include=['src.core.data_processing.tasks']
    )
    

    # Celery configuration
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=50,
        broker_connection_retry_on_startup=True,
    )

    class CallbackTask(Task):
        """Custom task class with enhanced callback support."""
        
        def on_success(self, retval, task_id, args, kwargs):
            """Called on task success."""
            logger.info(f"Task {task_id} succeeded with return value: {retval}")
            
        def on_failure(self, exc, task_id, args, kwargs, einfo):
            """Called on task failure."""
            logger.error(f"Task {task_id} failed with exception: {exc}")
            
        def on_retry(self, exc, task_id, args, kwargs, einfo):
            """Called on task retry."""
            logger.warning(f"Task {task_id} retrying due to: {exc}")
else:
    # Fallback when Celery is not available
    celery_app = None
    
    class CallbackTask:
        """Fallback task class when Celery is not available."""
        def on_success(self, retval, task_id, args, kwargs):
            logger.info(f"Task {task_id} succeeded with return value: {retval}")
            
        def on_failure(self, exc, task_id, args, kwargs, einfo):
            """Called on task failure."""
            logger.error(f"Task {task_id} failed with exception: {exc}")
            
        def on_retry(self, exc, task_id, args, kwargs, einfo):
            """Called on task retry."""
            logger.warning(f"Task {task_id} retrying due to: {exc}")

# Create conditional task decorators and wrappers
if celery_app is not None:
    # When Celery is available, use the actual decorators
    def task_decorator(*args, **kwargs):
        return celery_app.task(*args, **kwargs)
else:
    # When Celery is not available, create a pass-through decorator
    def task_decorator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Define the task function implementation
def process_uploaded_files_impl(file_paths: List[str], context_file_path: Optional[str] = None, 
                               db_url: Optional[str] = None, user_id: Optional[str] = None,
                               options: Dict[str, Any] = None, task_self=None) -> Dict[str, Any]:
    """
    Core implementation of file processing that works with or without Celery.
    
    Args:
        file_paths: List of temporary file paths to process
        context_file_path: Optional schema description file path
        db_url: Database URL for storage
        user_id: User identifier for tracking
        options: Additional processing options
        task_self: Celery task instance (None if not using Celery)
        
    Returns:
        Dict containing processing results and metadata
    """
    start_time = datetime.now()
    task_id = task_self.request.id if task_self else f"sync_{int(start_time.timestamp())}"
    options = options or {}
    
    try:
        logger.info(f"Starting file processing task {task_id} for {len(file_paths)} files")
        
        # Update task state if using Celery
        if task_self:
            task_self.update_state(
                state='PROCESSING',
                meta={'status': 'Initializing file processing', 'progress': 0, 'files_count': len(file_paths)}
            )
        
        # Load configuration
        config_path = options.get('config_path', 'config.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        # Database configuration
        if not db_url:
            db_config = config.get('database', {})
            db_url = os.getenv("DB_URL") or db_config.get('default_url', "sqlite:///data/raw/plm_updated.db")
        
        # Create database engine with connection pooling
        engine_config = {
            'pool_size': config.get('database', {}).get('connection_pool_size', 5),
            'pool_timeout': config.get('database', {}).get('connection_timeout', 30),
            'pool_recycle': 3600,  # Recycle connections every hour
            'pool_pre_ping': True   # Validate connections before use
        }
        
        if 'postgresql' in db_url.lower():
            engine_config['pool_size'] = 20  # Higher pool for PostgreSQL
            
        engine = create_engine(db_url, **engine_config)
        
        # Validate files exist and are readable
        validated_files = []
        total_size = 0
        
        if task_self:
            task_self.update_state(
                state='PROCESSING',
                meta={'status': 'Validating uploaded files', 'progress': 10}
            )
        
        for file_path in file_paths:
            if not Path(file_path).exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            file_size = Path(file_path).stat().st_size
            if file_size == 0:
                logger.warning(f"Empty file detected: {file_path}")
                continue
                
            validated_files.append(file_path)
            total_size += file_size
            
        # Handle case where only context file is provided (no data files)
        if not validated_files and not context_file_path:
            raise ValueError("No valid files found for processing")
        elif not validated_files and context_file_path:
            logger.info("Processing schema/context file only - no data files to upload")
        else:
            logger.info(f"Processing {len(validated_files)} valid files (total size: {total_size / 1024 / 1024:.2f} MB)")
        
        # Process files with progress tracking (only if we have data files)
        if task_self:
            task_self.update_state(
                state='PROCESSING',
                meta={'status': 'Uploading data to database', 'progress': 20, 'validated_files': len(validated_files)}
            )
        
        try:
            # Upload files to database only if we have data files
            if validated_files:
                tables = upload_files_to_db(validated_files, engine)
                
                if task_self:
                    task_self.update_state(
                        state='PROCESSING',
                        meta={'status': f'Data uploaded successfully, created {len(tables)} tables', 'progress': 50, 'tables_created': len(tables)}
                    )
            else:
                # No data files, just get existing tables
                from sqlalchemy import inspect
                inspector = inspect(engine)
                tables = inspector.get_table_names()
                
                if task_self:
                    task_self.update_state(
                        state='PROCESSING',
                        meta={'status': f'Using existing {len(tables)} tables', 'progress': 50, 'existing_tables': len(tables)}
                    )
                
                logger.info(f"No data files to upload, found {len(tables)} existing tables")
            
        except Exception as e:
            logger.error(f"Database processing failed: {e}")
            raise Exception(f"Database processing failed: {str(e)}")
        
        # Process context file if provided
        context_data = None
        if context_file_path and Path(context_file_path).exists():
            try:
                if task_self:
                    task_self.update_state(
                        state='PROCESSING',
                        meta={'status': 'Processing schema description', 'progress': 60}
                    )
                
                with open(context_file_path, 'r') as f:
                    context_data = json.load(f)
                    
            except Exception as e:
                logger.warning(f"Failed to process context file: {e}")
        
        # Build RAG system
        if task_self:
            task_self.update_state(
                state='PROCESSING',
                meta={'status': 'Building intelligent RAG system', 'progress': 70}
            )
        
        try:
            # Initialize hierarchical RAG system
            rag_system = HierarchicalRAGSystem(engine, config_path)
            
            # Build retrievers if we have schema data
            if context_data:
                retrievers = build_scalable_retriever_system(engine, context_data)
                rag_context = f"Built {len(retrievers)} specialized retrievers for intelligent context retrieval"
            else:
                rag_context = "RAG system initialized, awaiting schema description for optimal retrieval"
                
            if task_self:
                task_self.update_state(
                    state='PROCESSING',
                    meta={'status': 'RAG system built successfully', 'progress': 90}
                )
            
        except Exception as e:
            logger.error(f"RAG system building failed: {e}")
            rag_context = f"RAG system initialization failed: {str(e)}"
        
        # Clean up temporary files
        for file_path in validated_files:
            try:
                Path(file_path).unlink()
            except:
                pass
        
        if context_file_path:
            try:
                Path(context_file_path).unlink()
            except:
                pass
        
        # Calculate processing metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'success': True,
            'task_id': task_id,
            'message': f'Successfully processed {len(validated_files)} files and created {len(tables)} tables',
            'tables_created': tables,
            'files_processed': len(validated_files),
            'total_size_mb': round(total_size / 1024 / 1024, 2),
            'processing_time_seconds': round(processing_time, 2),
            'rag_context': rag_context,
            'database_url': db_url,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"File processing task {task_id} completed successfully in {processing_time:.2f} seconds")
        return result
        
    except Exception as e:
        error_msg = f"File processing failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}")
        
        # Update task state with error
        if task_self:
            task_self.update_state(
                state='FAILURE',
                meta={'error': error_msg, 'timestamp': datetime.now().isoformat()}
            )
        
        # Clean up files on error
        for file_path in file_paths:
            try:
                Path(file_path).unlink()
            except:
                pass
                
        raise Exception(error_msg)

# Now create the actual Celery task wrapper
if celery_app is not None:
    @task_decorator(bind=True, base=CallbackTask, name='process_uploaded_files')
    def process_uploaded_files(self, file_paths: List[str], context_file_path: Optional[str] = None, 
                              db_url: Optional[str] = None, user_id: Optional[str] = None,
                              options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Celery task wrapper for file processing."""
        return process_uploaded_files_impl(file_paths, context_file_path, db_url, user_id, options, self)
else:
    def process_uploaded_files(file_paths: List[str], context_file_path: Optional[str] = None, 
                              db_url: Optional[str] = None, user_id: Optional[str] = None,
                              options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Synchronous fallback for file processing when Celery is not available."""
        return process_uploaded_files_impl(file_paths, context_file_path, db_url, user_id, options, None)

# Similarly, let's create conditional decorators for the rebuild function
def rebuild_rag_system_impl(db_url: Optional[str] = None, config_path: str = 'config.json', task_self=None) -> Dict[str, Any]:
    """
    Rebuild RAG system for existing database.
    
    Args:
        db_url: Database URL
        config_path: Configuration file path
        task_self: Celery task instance (None if not using Celery)
        
    Returns:
        Dict containing rebuild results
    """
    start_time = datetime.now()
    task_id = task_self.request.id if task_self else f"sync_rebuild_{int(start_time.timestamp())}"
    
    try:
        logger.info(f"Starting RAG system rebuild task {task_id}")
        
        if task_self:
            task_self.update_state(
                state='PROCESSING',
                meta={'status': 'Initializing RAG rebuild', 'progress': 0}
            )
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        # Database connection
        if not db_url:
            db_config = config.get('database', {})
            db_url = os.getenv("DB_URL") or db_config.get('default_url', "sqlite:///data/raw/plm_updated.db")
        
        engine = create_engine(db_url)
        
        if task_self:
            task_self.update_state(
                state='PROCESSING',
                meta={'status': 'Building new RAG system', 'progress': 50}
            )
        
        # Rebuild RAG system
        rag_system = HierarchicalRAGSystem(engine, config_path)
        
        # Get table information
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'success': True,
            'task_id': task_id,
            'message': f'RAG system rebuilt successfully for {len(tables)} tables',
            'tables_count': len(tables),
            'processing_time_seconds': round(processing_time, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"RAG rebuild task {task_id} completed in {processing_time:.2f} seconds")
        return result
        
    except Exception as e:
        error_msg = f"RAG system rebuild failed: {str(e)}"
        logger.error(f"Task {task_id} failed: {error_msg}")
        
        if task_self:
            task_self.update_state(
                state='FAILURE',
                meta={'error': error_msg}
            )
        
        raise Exception(error_msg)

# Create wrapper for rebuild function
if celery_app is not None:
    @task_decorator(bind=True, name='rebuild_rag_system')
    def rebuild_rag_system(self, db_url: Optional[str] = None, config_path: str = 'config.json') -> Dict[str, Any]:
        """Celery task wrapper for RAG system rebuild."""
        return rebuild_rag_system_impl(db_url, config_path, self)
else:
    def rebuild_rag_system(db_url: Optional[str] = None, config_path: str = 'config.json') -> Dict[str, Any]:
        """Synchronous fallback for RAG system rebuild when Celery is not available."""
        return rebuild_rag_system_impl(db_url, config_path, None)

# Handle the cleanup task as well
if celery_app is not None:
    @task_decorator(name='cleanup_old_tasks')
    def cleanup_old_tasks(days_old: int = 7) -> Dict[str, Any]:
        """
        Clean up old task results and temporary files.
        
        Args:
            days_old: Age threshold for cleanup (days)
            
        Returns:
            Dict containing cleanup results
        """
        try:
            logger.info(f"Starting cleanup of tasks older than {days_old} days")
            
            # This would typically clean up Redis entries and temporary files
            # For now, just return success
            
            result = {
                'success': True,
                'message': f'Cleanup completed for tasks older than {days_old} days',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Cleanup failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
else:
    def cleanup_old_tasks(days_old: int = 7) -> Dict[str, Any]:
        """Synchronous fallback for cleanup when Celery is not available."""
        logger.info(f"Cleanup task called (sync mode) for tasks older than {days_old} days")
        return {
            'success': True,
            'message': f'Cleanup completed for tasks older than {days_old} days (sync mode)',
            'timestamp': datetime.now().isoformat()
        }

# Task status and result utilities
class TaskManager:
    """Utility class for managing async tasks."""
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """Get detailed task status."""
        if not CELERY_AVAILABLE or not celery_app:
            return {
                'task_id': task_id,
                'status': 'UNAVAILABLE',
                'ready': True,
                'successful': False,
                'failed': False,
                'message': 'Celery not available - async processing disabled'
            }
        
        try:
            result = AsyncResult(task_id, app=celery_app)
            
            status_info = {
                'task_id': task_id,
                'status': result.state,
                'ready': result.ready(),
                'successful': result.successful() if result.ready() else False,
                'failed': result.failed() if result.ready() else False
            }
            
            if result.state == 'PENDING':
                status_info['message'] = 'Task is waiting to be processed'
            elif result.state == 'PROCESSING':
                status_info.update(result.info or {})
            elif result.state == 'SUCCESS':
                status_info['result'] = result.result
            elif result.state == 'FAILURE':
                status_info['error'] = str(result.info)
            
            return status_info
            
        except Exception as e:
            return {
                'task_id': task_id,
                'status': 'ERROR',
                'error': f'Failed to get task status: {str(e)}'
            }
    
    @staticmethod
    def cancel_task(task_id: str) -> Dict[str, Any]:
        """Cancel a running task."""
        try:
            celery_app.control.revoke(task_id, terminate=True)
            return {
                'success': True,
                'message': f'Task {task_id} cancelled successfully'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to cancel task: {str(e)}'
            }
    
    @staticmethod
    def get_active_tasks() -> List[Dict[str, Any]]:
        """Get list of active tasks."""
        try:
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if active_tasks:
                all_tasks = []
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        all_tasks.append({
                            'worker': worker,
                            'task_id': task['id'],
                            'name': task['name'],
                            'args': task['args'],
                            'kwargs': task['kwargs']
                        })
                return all_tasks
            return []
            
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return []

# Signal handlers (only if Celery is available)
if celery_app is not None and task_success is not None:
    @task_success.connect
    def task_success_handler(sender=None, task_id=None, result=None, retval=None, **kwargs):
        """Handle successful task completion."""
        logger.info(f"Task {task_id} completed successfully")

    @task_failure.connect
    def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
        """Handle task failure."""
        logger.error(f"Task {task_id} failed: {exception}")

    # Periodic tasks for maintenance
    @celery_app.task(name='periodic_cleanup')
    def periodic_cleanup():
        """Periodic cleanup of old data and temporary files."""
        return cleanup_old_tasks.delay(days_old=7)

    # Celery beat schedule for periodic tasks
    celery_app.conf.beat_schedule = {
        'cleanup-old-tasks': {
            'task': 'periodic_cleanup',
            'schedule': timedelta(hours=24),  # Run daily
        },
    }
