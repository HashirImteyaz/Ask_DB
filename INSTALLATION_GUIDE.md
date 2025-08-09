# NLQ PLM System Installation Guide

This guide provides different installation options based on your needs and environment.

## Quick Start (Minimal Installation)

For the most stable installation, use the minimal requirements:

```bash
pip install -r requirements_minimal.txt
```

## Standard Installation

For full functionality:

```bash
pip install -r requirements.txt
```

## Manual Installation (Recommended for compatibility issues)

If you encounter compatibility issues, install packages in this order:

### Core Dependencies
```bash
pip install fastapi==0.104.1 uvicorn==0.24.0
pip install python-dotenv==1.0.0 python-multipart==0.0.6 PyYAML==6.0.1
pip install sqlalchemy==2.0.23
pip install requests==2.31.0 pydantic==2.4.2
```

### Data Processing
```bash
pip install pandas==2.0.3 numpy==1.24.4
pip install matplotlib==3.7.5
pip install scikit-learn==1.3.2
```

### AI Components
```bash
pip install openai==1.45.0 tiktoken==0.7.0
pip install langchain-core==0.2.39
pip install langchain-openai==0.1.25
pip install langgraph==0.0.26
```

### LlamaIndex (if needed for advanced RAG)
```bash
pip install llama-index-core==0.10.65
pip install llama-index-embeddings-openai==0.1.11
pip install llama-index-llms-openai==0.1.29
```

### Optional Components
```bash
# For UI
pip install streamlit==1.28.2

# For caching (optional)
pip install redis==4.6.0

# For SQL parsing (optional)
pip install sqlparse==0.4.4

# For background tasks (optional)
pip install celery==5.3.4

# For database support (optional)
pip install psycopg2-binary==2.9.7  # PostgreSQL
pip install PyMySQL==1.1.0          # MySQL
```

## Troubleshooting

### Common Issues:

1. **LlamaIndex Import Errors**: The system includes fallback imports, so LlamaIndex is optional
2. **Redis Connection Errors**: Redis caching is optional, system will work without it
3. **scikit-learn Issues**: Try installing with `pip install --no-cache-dir scikit-learn==1.3.2`
4. **Version Conflicts**: Use virtual environment and install minimal requirements first

### Virtual Environment Setup:
```bash
python -m venv nlq_env
# Windows
nlq_env\Scripts\activate
# Linux/Mac
source nlq_env/bin/activate

pip install --upgrade pip
pip install -r requirements_minimal.txt
```

### System Requirements:
- Python 3.8+
- 4GB RAM minimum
- Internet connection for OpenAI API

### Configuration:
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
3. Configure database URL if needed

### Running the System:
```bash
# API Server
python -m uvicorn src.api.main_app:app --reload

# Streamlit UI
streamlit run src/ui/streamlit_chat.py
```

## Features Available with Minimal Installation:
- ✅ Natural Language to SQL conversion
- ✅ Basic chat interface
- ✅ SQL execution and results
- ✅ Simple caching
- ✅ Error handling
- ❌ Advanced vector search (requires LlamaIndex)
- ❌ Redis caching (requires Redis)
- ❌ Background processing (requires Celery)

## Features Available with Full Installation:
- ✅ All minimal features
- ✅ Advanced vector search and RAG
- ✅ Redis-based caching
- ✅ Background task processing
- ✅ SQL optimization and validation
- ✅ Multiple database support
