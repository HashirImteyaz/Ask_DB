# 🚀 Quick Start Guide - NLQ PLM System

Get up and running with the Natural Language Query system in minutes!

## ⚡ 3-Minute Setup

### Step 1: Check Prerequisites
```bash
# Verify Python version (3.8+ required)
python --version

# Check if you have the database file
ls data/raw/plm_updated.db
```

### Step 2: Install Dependencies
```bash
# Install main dependencies
pip install -r src/config/requirements.txt

# Optional: Install Streamlit for chat UI
pip install -r src/config/streamlit_requirements.txt
```

### Step 3: Start the System

#### Option A: HTML Interface (Recommended for first time)
```bash
# Start API server
python scripts/start_api.py

# Open browser to: http://127.0.0.1:8000
```

#### Option B: Streamlit Chat Interface
```bash
# Terminal 1: Start API
python scripts/start_api.py

# Terminal 2: Start Streamlit (in new terminal)
python scripts/start_streamlit.py

# Open browser to: http://localhost:8501
```

## 🎯 First Queries to Try

Once your system is running, try these example queries:

### 📝 Basic Queries
```
1. "List all recipes"
2. "Show me ingredients from Europe"  
3. "What recipes contain chicken?"
4. "Count total ingredients"
```

### 📊 Chart Queries
```
1. "Create a bar chart of top 10 ingredients"
2. "Show me a pie chart of recipes by region"
3. "Display ingredient distribution as a graph"
```

### 🌍 Geographic Queries
```
1. "Show ingredients from France"
2. "List recipes manufactured in India"
3. "Products from Asia business unit"
```

## 🔧 Troubleshooting

### API Won't Start
```bash
# Check if port 8000 is available
netstat -an | grep 8000

# Try different port
python scripts/start_api.py --port 8001
```

### Database Not Found
```bash
# Check database location
ls data/raw/
# Should see: plm_updated.db

# If missing, check DATA/ folder for original files
ls DATA/
```

### Import Errors
```bash
# Install missing packages
pip install -r src/config/requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Streamlit Issues
```bash
# Install Streamlit
pip install streamlit>=1.28.0

# Clear Streamlit cache
streamlit cache clear
```

## 📱 Interface Guide

### HTML Interface Features
- **Upload Area**: Drag & drop data files
- **Chat Box**: Type natural language queries
- **SQL Display**: See generated SQL queries
- **Tables**: View results in formatted tables
- **Charts**: Automatic visualizations

### Streamlit Interface Features  
- **Sidebar Controls**: API status, clear chat
- **Chat History**: Persistent conversation
- **Statistics**: Message and query counts
- **Real-time Updates**: Live API status

## 🎨 Customization

### Change Database Path
Edit `src/api/main_app.py`:
```python
DB_URL = "sqlite:///your/database/path.db"
```

### Modify Schema
Edit `src/config/schema_description.json` to match your data structure.

### Add Custom Prompts
Update `src/core/agent/prompts.py` for custom query patterns.

## 📊 Expected Output Format

Every query returns:
```
✅ Natural Language Answer
🔍 Generated SQL Query  
📋 Data Table (if applicable)
📈 Chart/Graph (if requested)
📊 Summary Statistics
```

## 🆘 Getting Help

### Check Logs
```bash
# API logs appear in terminal
# Look for ERROR or WARNING messages

# For detailed debugging:
export LOG_LEVEL=DEBUG
python scripts/start_api.py
```

### Validate Setup
```bash
# Test API health
curl http://127.0.0.1:8000/health

# Should return: {"status": "healthy", ...}
```

### Common Issues

1. **"Agent not available"** → Restart API server
2. **"Database not found"** → Check data/raw/ folder
3. **"Import errors"** → Install requirements.txt
4. **"Port in use"** → Change port or kill existing process

## 🎉 You're Ready!

Your NLQ PLM system is now running. Start with simple queries and gradually explore more complex analytical questions. The system learns from your queries and provides increasingly better responses!

### Next Steps
1. Try the example queries above
2. Explore your specific data with custom questions
3. Experiment with chart generation
4. Check out the full documentation in `docs/`

---
**Happy Querying! 🚀**