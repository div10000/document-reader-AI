import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from backend import extract_pages, build_bm25_index, get_answer
from vision_backend import ask_image  # --- NEW: Import our vision module ---

# Load the API key from the .env file
load_dotenv()

# Initialize the Groq LLM
@st.cache_resource
def get_llm():
    # It automatically picks up GROQ_API_KEY from the environment
    return ChatGroq(model_name="openai/gpt-oss-120b")

llm = get_llm()

# --- App Title & Mode Selector ---
st.title("🤖 Multi-Modal AI Assistant")

with st.sidebar:
    st.header("Select Mode")
    app_mode = st.radio("What do you want to chat with?", ["📄 Document Search", "🖼️ Image Analysis"])
    st.markdown("---")

# ==========================================
# MODE 1: DOCUMENT SEARCH (Your exact previous code)
# ==========================================
if app_mode == "📄 Document Search":
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
        # Automatically create the folder if it doesn't exist
        os.makedirs("temp_uploads", exist_ok=True)
        
        # Save the uploaded file temporarily
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text and build the search index
        with st.spinner("Processing document..."):
            # We check current_file so it knows to rebuild if you upload a NEW document
            if "pages" not in st.session_state or st.session_state.get("current_file") != uploaded_file.name:
                st.session_state.pages = extract_pages(file_path)
                st.session_state.bm25_index = build_bm25_index(st.session_state.pages)
                st.session_state.current_file = uploaded_file.name
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
                    
                    # Grab the last 4 messages 
                    recent_history = st.session_state.messages[-5:-1]
                    
                    # Pass recent_history to the backend
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

# ==========================================
# MODE 2: IMAGE ANALYSIS (The new feature!)
# ==========================================
elif app_mode == "🖼️ Image Analysis":
    st.write("Upload an image and ask questions about what's inside it!")
    
    with st.sidebar:
        st.header("1. Upload Image")
        uploaded_img = st.file_uploader("Choose an Image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_img is not None:
        # Show the image on the screen
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        
        st.header("2. Chat")
        
        # Initialize a SEPARATE chat history just for images
        if "img_messages" not in st.session_state:
            st.session_state.img_messages = []

        for msg in st.session_state.img_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_prompt := st.chat_input("Ask a question about this image..."):
            st.chat_message("user").markdown(user_prompt)
            st.session_state.img_messages.append({"role": "user", "content": user_prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing image..."):
                    # Call the vision backend from the separate file
                    answer = ask_image(user_prompt, uploaded_img)
                    st.markdown(answer)
                    
            st.session_state.img_messages.append({"role": "assistant", "content": answer})