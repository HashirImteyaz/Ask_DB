@echo off
echo Starting Streamlit Chat Interface...
echo.
echo Make sure your main API is running on http://127.0.0.1:8000
echo.
pause
streamlit run streamlit_chat.py --server.port 8501 --server.address localhost