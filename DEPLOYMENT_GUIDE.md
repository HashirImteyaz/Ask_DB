# NLQ PLM System - Complete Deployment and Testing Guide

## 🚀 System Overview

The Enhanced NLQ PLM (Natural Language Query - Product Lifecycle Management) System is now a **truly powerful, intelligent, and robust** system designed to handle any scenario and query on both small and huge databases efficiently and accurately.

### ✨ Key Enhancements Implemented

1. **🔧 Scalability Infrastructure**
   - Asynchronous processing with Celery
   - Multi-level Redis caching system
   - Database connection pooling
   - PostgreSQL/MySQL support for large-scale deployments

2. **🛡️ SQL Intelligence & Security**
   - Advanced SQL validation and optimization
   - Security vulnerability detection
   - Performance analysis and suggestions
   - Database statistics collection

3. **💰 Cost Optimization**
   - Intelligent model routing (GPT-4 vs GPT-3.5-turbo)
   - Multi-level result caching
   - Context reuse and optimization
   - Detailed cost tracking and budgeting

4. **🧠 Enhanced Agent Intelligence**
   - Hierarchical RAG system for better context
   - Conversation memory with vector search
   - Query classification and routing
   - Session-based learning and adaptation

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux Ubuntu 18.04+
- **Python**: 3.9+ (3.11 recommended)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large databases)
- **Storage**: 10GB+ available disk space
- **Network**: Internet connection for LLM API calls

### Required Services
- **Redis Server** (for caching and async processing)
- **PostgreSQL** (optional, for production deployments)
- **OpenAI API Key** (required for LLM functionality)

---

## 🛠️ Installation Guide

### Step 1: Clone and Set Up Environment

```powershell
# Clone the repository
git clone <repository-url>
cd NLQ_PLM-main

# Create virtual environment
python -m venv nlq_env
nlq_env\Scripts\activate  # On Windows
# source nlq_env/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements_enhanced.txt
```

### Step 2: Install and Start Redis

#### Windows - Option 1: Using Chocolatey (Recommended)
**Prerequisites**: Run PowerShell as Administrator

```powershell
# Right-click PowerShell and select "Run as Administrator"
# Then run the following commands:

# Install Chocolatey (if not installed)
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Close and reopen PowerShell as Administrator, then:
choco install redis-64

# Start Redis
redis-server
```

#### Windows - Option 2: Manual Installation (No Admin Required) ✅ COMPLETED
If you've followed the manual setup, Redis is already installed and ready to use:

```powershell
# Start Redis (use the provided batch file)
.\start_redis.bat

# Or start manually:
cd redis
.\redis-server.exe --port 6379

# Test Redis in a new terminal:
.\test_redis.bat

# Or test manually:
cd redis
.\redis-cli.exe ping
# Should respond with "PONG"
```

**Note**: The Redis server is already downloaded and extracted in your project folder at `./redis/`

#### Windows - Option 3: Using Windows Subsystem for Linux (WSL)
```bash
# Install WSL2 if not already installed
# Then run in WSL terminal:
sudo apt update
sudo apt install redis-server
redis-server
```

#### macOS
```bash
# Install using Homebrew
brew install redis

# Start Redis
brew services start redis
```

#### Linux (Ubuntu/Debian)
```bash
# Install Redis
sudo apt update
sudo apt install redis-server

# Start Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration (SQLite default, PostgreSQL for production)
DATABASE_URL=sqlite:///DATA/plm_updated.db
# DATABASE_URL=postgresql://user:password@localhost:5432/nlq_plm

# Redis Configuration
REDIS_URL=redis://localhost:6379/1
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# Feature Flags
ASYNC_PROCESSING_ENABLED=true
CACHING_ENABLED=true
SQL_VALIDATION_ENABLED=true
DATABASE_STATISTICS_ENABLED=true

# Logging
LOG_LEVEL=INFO
```

### Step 4: Set Up PostgreSQL (Production) ✅ POSTGRESQL INSTALLED

Since you have PostgreSQL installed, let's configure it for the NLQ PLM system:

#### Step 4a: Connect to PostgreSQL
```powershell
# Open Command Prompt or PowerShell as Administrator
# Connect to PostgreSQL (replace 'postgres' with your superuser if different)
psql -U postgres
```

If you get a "command not found" error, you may need to add PostgreSQL to your PATH or use the full path:
```powershell
# Example full path (adjust version number as needed):
"C:\Program Files\PostgreSQL\15\bin\psql.exe" -U postgres
```

#### Step 4b: Create Database and User
Once connected to PostgreSQL, run these SQL commands:

**Important**: You should see `postgres=#` prompt. If you see `postgres-#`, press `Ctrl+C` to cancel the current command and start fresh.

```sql
-- Create the database for NLQ PLM
CREATE DATABASE nlq_plm;

-- Create a dedicated user for the application
CREATE USER nlq_user WITH ENCRYPTED PASSWORD 'nlq_secure_password_2024';

-- Grant all privileges on the database to the user
GRANT ALL PRIVILEGES ON DATABASE nlq_plm TO nlq_user;

-- Connect to the newly created database
\c nlq_plm

-- Grant schema privileges (run these after connecting to nlq_plm database)
GRANT ALL ON SCHEMA public TO nlq_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO nlq_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO nlq_user;

-- Optional: Grant default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO nlq_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO nlq_user;

-- Verify the setup (optional)
\l
\du

-- Exit PostgreSQL
\q
```

#### Step 4c: Test the Connection
```powershell
# Test connecting with the new user
psql -U nlq_user -d nlq_plm -h localhost

# You should be prompted for the password: sa
# If successful, you'll see the database prompt: nlq_plm=>
# Exit with: \q
```

#### Step 4d: Update Environment Configuration
Update your `.env` file to use PostgreSQL instead of SQLite:

```env
# PostgreSQL Configuration (replace SQLite)
DATABASE_URL=postgresql://nlq_user:sa@localhost:5432/nlq_plm

# Keep other settings the same
REDIS_URL=redis://localhost:6379/1
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

---

## 🚦 Running the System

### Step 1: Start Redis (if not running)
```powershell
redis-server
```

### Step 2: Start Celery Worker (for async processing)
```powershell
# In a new terminal window
cd NLQ_PLM-main
venv\Scripts\activate  # Activate your virtual environment
python -m celery -A src.core.data_processing.tasks worker --loglevel=info
```

**Alternative methods if the above doesn't work:**
```powershell
# Method 1: Use python -m celery
python -m celery -A src.core.data_processing.tasks worker --loglevel=info

# Method 2: Use full path to celery executable
venv\Scripts\celery.exe -A src.core.data_processing.tasks worker --loglevel=info

# Method 3: If Celery is not installed
pip install celery[redis]==5.3.4
```

### Step 3: Start the Main Application
```powershell
# In the main terminal
python -m src.api.main_app
```

Or using uvicorn directly:
```powershell
uvicorn src.api.main_app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 4: Access the System

- **API Documentation**: http://localhost:8000/docs
- **Web Interface**: http://localhost:8000/
- **Health Check**: http://localhost:8000/health

---

## 🧪 Testing Guide

### 1. Health Check
```powershell
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "celery": "active",
  "features": {
    "async_processing": true,
    "caching": true,
    "sql_validation": true,
    "database_statistics": true
  }
}
```

### 2. Upload Test Data

#### Using Web Interface
1. Navigate to http://localhost:8000/
2. Click "Upload Files"
3. Upload your Excel/CSV files
4. Wait for processing confirmation

#### Using API
```powershell
curl -X POST "http://localhost:8000/upload/" -H "Content-Type: multipart/form-data" -F "files=@your_file.xlsx"
```

### 3. Test Natural Language Queries

#### Simple Queries
```json
POST http://localhost:8000/query/
{
  "query": "Show me all products",
  "session_id": "test_session_1"
}
```

#### Complex Analytical Queries
```json
POST http://localhost:8000/query/
{
  "query": "What are the top 5 products by revenue in the last quarter, and show me the trend analysis",
  "session_id": "test_session_1"
}
```

#### Aggregation and Grouping
```json
POST http://localhost:8000/query/
{
  "query": "Group products by category and show average price, total sales, and profit margin for each category",
  "session_id": "test_session_1"
}
```

### 4. Test Advanced Features

#### Cache Performance
```json
# First query (will be cached)
POST http://localhost:8000/query/
{
  "query": "Show product sales summary",
  "session_id": "cache_test"
}

# Same query again (should return from cache)
POST http://localhost:8000/query/
{
  "query": "Show product sales summary",
  "session_id": "cache_test"
}
```

#### SQL Validation
```json
GET http://localhost:8000/system/validate-sql?sql=SELECT * FROM products WHERE price > 100
```

#### System Statistics
```json
GET http://localhost:8000/system/stats
```

---

## 🔍 Monitoring and Administration

### System Health Monitoring
```json
GET http://localhost:8000/system/health
```

### Cache Statistics
```json
GET http://localhost:8000/system/cache-stats
```

### Usage Analytics
```json
GET http://localhost:8000/system/usage-stats
```

### Clear Cache (if needed)
```json
POST http://localhost:8000/system/clear-cache
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Redis Installation Issues

**Problem**: Chocolatey installation fails with "Administrative permissions" error

**Solutions**:
- **Option A**: Run PowerShell as Administrator
  1. Right-click PowerShell in Start menu
  2. Select "Run as administrator"
  3. Run the Chocolatey installation command
  
- **Option B**: Manual Redis installation (No admin required)
  1. Download Redis from: https://github.com/microsoftarchive/redis/releases
  2. Download latest .zip file (e.g., Redis-x64-3.0.504.zip)
  3. Extract to folder like `C:\redis` or `C:\Users\YourName\redis`
  4. Navigate to Redis folder: `cd C:\path\to\redis`
  5. Start Redis: `.\redis-server.exe`
  6. Test in new terminal: `.\redis-cli.exe ping` (should return "PONG")

- **Option C**: Use Docker (if Docker Desktop is installed)
  ```powershell
  docker run -d -p 6379:6379 --name redis redis:alpine
  ```

- **Option D**: Use Windows Subsystem for Linux (WSL)
  ```bash
  # In WSL terminal:
  sudo apt update && sudo apt install redis-server
  redis-server
  ```

#### 2. Redis Connection Error
**Problem**: `ConnectionError: Error 10061 connecting to localhost:6379`

**Solutions**:
- Ensure Redis server is running: `redis-server`
- Check Redis configuration in `.env` file
- For Windows, use Redis for Windows or WSL

#### 2. Database Connection Issues
**Problem**: `Database connection failed`

**Solutions**:
- Check database URL in `.env` file
- Ensure database file exists (for SQLite)
- Verify PostgreSQL server is running (for PostgreSQL)
- Check database credentials and permissions

#### 3. OpenAI API Errors
**Problem**: `InvalidAPIKey` or `RateLimitExceeded`

**Solutions**:
- Verify `OPENAI_API_KEY` in `.env` file
- Check API key validity on OpenAI dashboard
- Monitor usage limits and billing

#### 4. Celery Worker Not Starting
**Problem**: `Celery worker fails to start` or `'celery' is not recognized`

**Solutions**:
- **Install Celery**: `pip install celery[redis]==5.3.4`
- **Use python -m celery**: `python -m celery -A src.core.data_processing.tasks worker --loglevel=info`
- **Use full path**: `venv\Scripts\celery.exe -A src.core.data_processing.tasks worker --loglevel=info`
- Ensure Redis is running
- Check `CELERY_BROKER_URL` configuration
- Make sure you're in the correct virtual environment

#### 6. Network Connectivity Issues
**Problem**: `Failed to resolve 'openaipublic.blob.core.windows.net'` or similar network errors

**Solutions**:
- **Check Internet Connection**: Ensure you have a stable internet connection
- **Check Corporate Firewall**: If behind a corporate firewall, whitelist:
  - `openaipublic.blob.core.windows.net`
  - `api.openai.com`
- **DNS Issues**: Try using Google DNS (8.8.8.8, 8.8.4.4) or Cloudflare DNS (1.1.1.1)
- **VPN/Proxy**: If using VPN or proxy, ensure it allows HTTPS traffic to OpenAI domains
- **Offline Mode**: The system will work with limited functionality using fallback token counting

#### 7. Configuration Issues
**Problem**: `Missing configuration section: llm_settings` or similar warnings

**Solutions**:
- Configuration has been updated automatically
- Restart the application after seeing these warnings
- Check that `config.json` contains all required sections

### Debug Mode
Enable detailed logging by setting in `.env`:
```env
LOG_LEVEL=DEBUG
```

---

## 📈 Performance Optimization

### For Small Databases (< 1GB)
- Use SQLite with default configuration
- Enable in-memory caching
- Use cost-effective models for simple queries

### For Medium Databases (1-10GB)
- Switch to PostgreSQL
- Enable Redis caching
- Use connection pooling
- Enable async processing

### For Large Databases (> 10GB)
- Use PostgreSQL with optimized settings
- Enable all caching layers
- Increase connection pool size
- Use dedicated Redis instance
- Consider read replicas

### Configuration for Large Scale
```json
{
  "database": {
    "type": "postgresql",
    "pool_size": 50,
    "pool_timeout": 60,
    "pool_recycle_hours": 2
  },
  "scalability": {
    "caching": {
      "enabled": true,
      "compression_enabled": true,
      "cache_layers": ["sql_results", "llm_responses", "final_responses", "rag_contexts"]
    },
    "async_processing": {
      "max_concurrent_tasks": 10
    }
  }
}
```

---

## 🎯 Testing Different Query Scenarios

### 1. Basic Data Retrieval
- "Show me all products"
- "List customers from New York"
- "Display order details for order ID 12345"

### 2. Analytical Queries
- "What are the top 10 bestselling products?"
- "Show monthly sales trends for the last year"
- "Which customers have the highest lifetime value?"

### 3. Complex Aggregations
- "Group products by category and show total revenue for each"
- "Compare quarterly performance year over year"
- "Show customer segmentation based on purchase behavior"

### 4. Time-based Analysis
- "Show sales performance for the current quarter"
- "What was our revenue growth in the last 6 months?"
- "Display seasonal trends in product sales"

### 5. Multi-table Joins
- "Show customer names with their recent orders"
- "List products with supplier information"
- "Display sales rep performance with customer details"

---

## 🌟 Advanced Features

### 1. Conversational Context
The system maintains conversation history:
```json
{
  "query": "Show me product sales",
  "session_id": "conv_1"
}

# Follow-up query in same session
{
  "query": "Now show me just the top 5",
  "session_id": "conv_1"
}
```

### 2. Cost Optimization
Automatically routes queries to appropriate models:
- Simple queries → GPT-3.5-turbo (cost-effective)
- Complex analysis → GPT-4 (high accuracy)

### 3. Intelligent Caching
- SQL result caching
- LLM response caching
- Context caching for conversations
- Automatic cache invalidation

### 4. Security Features
- SQL injection detection
- Query validation
- Access pattern monitoring
- Audit logging

---

## 📊 Success Metrics

After deployment, monitor these metrics:

1. **Response Time**:
   - Target: < 2 seconds for cached queries
   - Target: < 10 seconds for complex new queries

2. **Cache Hit Rate**:
   - Target: > 80% for repeated queries

3. **Cost Efficiency**:
   - Track cost per query
   - Monitor model usage distribution

4. **Accuracy**:
   - Test with known query-result pairs
   - Monitor user feedback

5. **System Health**:
   - Database connection stability
   - Redis availability
   - Celery task completion rate

---

## 🎉 Congratulations!

You now have a **truly powerful, intelligent, and robust** NLQ PLM system that can:

✅ Handle any scenario and query efficiently  
✅ Scale from small to huge databases  
✅ Provide cost-effective operations  
✅ Maintain high accuracy and performance  
✅ Support production workloads  

For additional support or advanced configurations, refer to the detailed documentation in the `/docs` folder or reach out to the development team.

---

## 📚 Additional Resources

- **API Documentation**: Access `/docs` endpoint when running
- **Configuration Reference**: See `config.json` for all settings
- **Development Guide**: Check `/docs/development.md`
- **Troubleshooting**: See `/docs/troubleshooting.md`
