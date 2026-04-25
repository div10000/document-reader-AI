import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from backend import extract_pages, build_bm25_index, get_answer

# Load the API key from the .env file
load_dotenv()

# Initialize the Groq LLM
@st.cache_resource
def get_llm():
    # It automatically picks up GROQ_API_KEY from the environment
    return ChatGroq(model_name="openai/gpt-oss-120b")

llm = get_llm()

# --- UI Setup ---
st.title("📄 BM25 Document Chatbot")
st.write("Upload a document and ask questions about it!")

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a Document", 
        type=["pdf", "docx", "xlsx", "xls", "txt"]
    )

# Process the file if uploaded
if uploaded_file is not None:
    # --- NEW: Automatically create the folder if it doesn't exist ---
    os.makedirs("temp_uploads", exist_ok=True)
    
    # Save the uploaded file temporarily
    file_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text and build the search index (cached so it doesn't rebuild on every chat message)
    with st.spinner("Processing document..."):
        if "pages" not in st.session_state:
            st.session_state.pages = extract_pages(file_path)
            st.session_state.bm25_index = build_bm25_index(st.session_state.pages)
            st.success("Document processed! You can now ask questions.")

# --- Chat Interface ---
st.header("2. Chat")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_question := st.chat_input("Ask a question about your document..."):
    
    # 1. Show user message and save it
    st.chat_message("user").markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # 2. Check if document is uploaded
    if "bm25_index" not in st.session_state:
        st.error("Please upload a document first, bro!")
    else:
        # 3. Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching and thinking..."):
                
                # --- NEW: Grab the last 4 messages (excluding the one the user JUST typed) ---
                # Using Python list slicing: -5 to -1 gets the 4 items right before the newest one
                recent_history = st.session_state.messages[-5:-1]
                
                # --- NEW: Pass recent_history to the backend ---
                answer, sources = get_answer(
                    user_question, 
                    st.session_state.bm25_index, 
                    st.session_state.pages, 
                    llm,
                    recent_history
                )
                
                full_response = f"{answer}\n\n*(Scanned Pages: {', '.join(map(str, sources))})*"
                st.markdown(full_response)
                
        # 4. Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})