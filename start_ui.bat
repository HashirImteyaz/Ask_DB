@echo off
echo Starting Ask DB UI...
echo.
echo Starting API server...
start "API Server" python -c "from src.api.main_app import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8000, log_level='info')"

echo Waiting for API to start...
timeout /t 5 /nobreak > nul

echo Opening UI in browser...
start "Ask DB UI" "src\ui\index_improved.html"

echo.
echo ================================================================
echo   Ask DB UI is now running!
echo ================================================================
echo   API Server: http://127.0.0.1:8000
echo   UI: Opening in your default browser
echo.
echo   Features:
echo   ✓ Enhanced Multiple Retrieval System
echo   ✓ Column Description Retriever
echo   ✓ Table Description Retriever  
echo   ✓ Intelligent Query Analysis
echo   ✓ Advanced SQL Generation
echo.
echo   To test the multiple retrieval system, try queries like:
echo   - "What columns are available in the recipes table?"
echo   - "Show me the table structure for specifications"
echo   - "What data types are used in the database?"
echo.
echo   Press Ctrl+C in the API Server window to stop the server
echo ================================================================
pause
