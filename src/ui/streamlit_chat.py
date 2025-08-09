import streamlit as st
import requests
import json
import base64
from typing import Dict, List, Optional
import time

# Load configuration
try:
    with open("config.json", 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {}

# Page configuration with config values
ui_config = CONFIG.get('ui', {})
st.set_page_config(
    page_title="NLQ System Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration from config
api_config = CONFIG.get('api', {})
host = api_config.get('host', '127.0.0.1')
port = api_config.get('port', 8000)
API_BASE_URL = f"http://{host}:{port}"

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #e7f3ff;
        border-left-color: #2196F3;
    }
    .assistant-message {
        background-color: #f8f9ff;
        border-left-color: #667eea;
    }
    .sql-block {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
        white-space: pre-wrap;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_chat_history" not in st.session_state:
    st.session_state.api_chat_history = []

def check_api_health() -> bool:
    """Check if the API is running and healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_chat_message(message: str) -> Dict:
    """Send a chat message to the API"""
    try:
        payload = {
            "query": message,
            "history": st.session_state.api_chat_history
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"API Error: {response.status_code} - {response.text}",
                "sql": None,
                "graph": None
            }
    except requests.exceptions.Timeout:
        return {
            "answer": "Request timed out. Please try again.",
            "sql": None,
            "graph": None
        }
    except Exception as e:
        return {
            "answer": f"Error connecting to API: {str(e)}",
            "sql": None,
            "graph": None
        }

def clear_chat():
    """Clear all chat history"""
    st.session_state.chat_history = []
    st.session_state.api_chat_history = []

def upload_files(data_files, context_file=None):
    """Upload data files and optional context file to the API"""
    try:
        # Prepare files for upload
        files = []
        
        # Add data files
        if data_files:
            for file in data_files:
                files.append(("files", (file.name, file.getvalue(), file.type)))
        
        # Add context file if provided
        if context_file:
            files.append(("context_file", (context_file.name, context_file.getvalue(), context_file.type)))
        
        # Send upload request
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            return {"success": True, "message": response.json().get("message", "Upload successful")}
        else:
            return {"success": False, "message": f"Upload failed: {response.status_code} - {response.text}"}
    
    except Exception as e:
        return {"success": False, "message": f"Upload error: {str(e)}"}

def format_sql(sql: str) -> str:
    """Format SQL for better display"""
    if not sql:
        return ""
    
    # Add line breaks before major SQL keywords
    keywords = ['FROM', 'WHERE', 'JOIN', 'ORDER BY', 'GROUP BY', 'HAVING', 'LIMIT', 'UNION']
    formatted_sql = sql
    
    for keyword in keywords:
        formatted_sql = formatted_sql.replace(f' {keyword} ', f'\n{keyword} ')
        formatted_sql = formatted_sql.replace(f' {keyword.lower()} ', f'\n{keyword} ')
    
    return formatted_sql.strip()

def display_message(role: str, content: str, sql: Optional[str] = None, graph: Optional[str] = None):
    """Display a chat message with proper styling"""
    
    # Message container
    message_class = "user-message" if role == "user" else "assistant-message"
    
    with st.container():
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <strong>{"You" if role == "user" else "Assistant"}:</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the main content
        if content:
            st.markdown(content, unsafe_allow_html=True)
        
        # Display SQL query if present
        if sql:
            st.markdown("**Generated SQL Query:**")
            formatted_sql = format_sql(sql)
            st.markdown(f"""
            <div class="sql-block">{formatted_sql}</div>
            """, unsafe_allow_html=True)
        
        # Display graph if present
        if graph:
            try:
                image_data = base64.b64decode(graph)
                st.image(image_data, caption="Generated Visualization", use_column_width=True)
            except Exception as e:
                st.error(f"Error displaying graph: {e}")

# Main UI
def main():
    st.title("💬 NLQ System Chat Interface")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # API Status
        st.subheader("🔌 API Status")
        if check_api_health():
            st.markdown('<div class="success-message">✅ API is running</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">❌ API is not accessible</div>', unsafe_allow_html=True)
            st.error(f"Make sure the API is running on {API_BASE_URL}")
        
        st.divider()
        
        # File Upload Section
        st.subheader("📁 Upload Files")
        
        # Data files upload
        data_files = st.file_uploader(
            "Data Files",
            accept_multiple_files=True,
            type=['csv', 'xlsx', 'json', 'yaml', 'yml'],
            help="Upload your data files (CSV, Excel, JSON, YAML)"
        )
        
        # Context file upload
        context_file = st.file_uploader(
            "Context File (Optional)",
            accept_multiple_files=False,
            type=['json', 'yaml', 'yml', 'txt'],
            help="Upload business context/schema description file"
        )
        
        # Upload button
        if st.button("📤 Upload Files", type="primary", disabled=not data_files):
            if data_files:
                with st.spinner("Uploading files..."):
                    result = upload_files(data_files, context_file)
                    
                    if result["success"]:
                        st.success(result["message"])
                        # Clear chat history when new files are uploaded
                        clear_chat()
                        st.rerun()
                    else:
                        st.error(result["message"])
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            clear_chat()
            st.success("Chat history cleared!")
            st.rerun()
        
        st.divider()
        
        # Chat statistics
        st.subheader("📊 Chat Statistics")
        st.metric("Total Messages", len(st.session_state.chat_history))
        st.metric("API Messages", len(st.session_state.api_chat_history))
        
        st.divider()
        
        # Instructions
        st.subheader("💡 Instructions")
        st.markdown("""
        1. Type your query in the chat input below
        2. The system will generate SQL and provide answers
        3. You can ask about:
           - Recipe ingredients
           - Product specifications
           - Manufacturing locations
           - Data analysis queries
        4. Use the Clear button to start fresh
        """)

    # Main chat area
    st.subheader("Chat with the NLQ System")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Conversation History")
        
        for message in st.session_state.chat_history:
            display_message(
                role=message["role"],
                content=message["content"],
                sql=message.get("sql"),
                graph=message.get("graph")
            )
    else:
        st.info("👋 Welcome! Start by asking a question about your data.")
    
    # Chat input
    st.markdown("### Ask a Question")
    
    # Create two columns for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="e.g., List all recipes with chicken ingredients",
            key="user_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send 📤", type="primary")
    
    # Handle message sending
    if send_button and user_input.strip():
        # Add user message to history
        user_message = {
            "role": "user",
            "content": user_input,
            "sql": None,
            "graph": None
        }
        st.session_state.chat_history.append(user_message)
        st.session_state.api_chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Show loading spinner
        with st.spinner("🤔 Thinking..."):
            # Send to API
            api_response = send_chat_message(user_input)
            
            # Add assistant response to history
            assistant_message = {
                "role": "assistant",
                "content": api_response.get("answer", "No response received"),
                "sql": api_response.get("sql"),
                "graph": api_response.get("graph")
            }
            st.session_state.chat_history.append(assistant_message)
            st.session_state.api_chat_history.append({
                "role": "assistant",
                "content": api_response.get("answer", "No response received")
            })
        
        # Rerun to show new messages
        st.rerun()
    
    elif send_button and not user_input.strip():
        st.warning("Please enter a question before sending.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**NLQ System** | Built with Streamlit | "
        f"API Endpoint: `{API_BASE_URL}`"
    )

if __name__ == "__main__":
    main()