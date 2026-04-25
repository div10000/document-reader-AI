# 🚀 BM25 Document Chatbot 

A blazing-fast, lightweight document Q&A chatbot built with Python, Streamlit, and the Groq API. 

Unlike traditional LLM wrappers that struggle with rate limits, or heavy RAG pipelines that require complex vector databases, this project uses **BM25** (a battle-tested lexical search algorithm) to run local, zero-cost, and instant document retrieval.

## 💡 Why This Approach over Traditional RAG (Vector Embeddings)?

When building LLM apps over custom data, the industry default is Vector RAG (Embedding text -> Storing in Chroma/Pinecone -> Cosine Similarity Search). However, this project intentionally avoids that for several key advantages:

1. **Zero Vector Database Overhead:** No need to spin up, configure, or pay for external databases like Pinecone or run memory-heavy local vector stores. 
2. **No Embedding API Latency:** We don't have to send our document to an OpenAI/HuggingFace API just to convert text into numbers. BM25 runs entirely locally on your CPU in milliseconds.
3. **Exact Keyword Precision:** Vector embeddings often struggle with exact matches (like specific part numbers, acronyms, or names) because they look for "semantic meaning". BM25 is a statistical keyword algorithm, meaning if you search for a specific term, it finds the exact page containing that term.
4. **Token & Rate Limit Efficient:** By running the search locally and only passing the top 3 most relevant pages to the Groq API, we comfortably bypass strict Tokens-Per-Minute (TPM) limits while giving the LLM the exact context it needs.

## ⚙️ Workflow: Under the Hood



Here is exactly what happens when you interact with the app:

1. **Document Upload:** The user uploads a file via the Streamlit UI. Supported formats: `.pdf`, `.docx`, `.xlsx`, `.xls`, `.txt`.
2. **Intelligent Parsing & Chunking:**
   * **PDFs:** Split page-by-page.
   * **Word Docs:** Paragraphs and tables are extracted and grouped into 20-block "Sections".
   * **Excel:** Each sheet is parsed into a readable text table.
3. **Local Indexing:** The text is tokenized (split into words) and fed into the `rank_bm25` Okapi algorithm. This creates a lightweight, temporary search index in the app's memory.
4. **User Query & Retrieval:** When the user asks a question, the query is tokenized. The BM25 algorithm instantly calculates which 3 chunks (pages/sections) mathematically contain the most relevant keywords.
5. **LLM Generation:** The chatbot sends *only* those 3 relevant chunks + the user's prompt to Groq (Llama 3). The LLM synthesizes the final answer and cites its sources.

## 📂 Project Structure

```text
my_bm25_bot/
│
├── .env                 # Environment variables (Groq API Key)
├── app.py               # Streamlit frontend (UI, chat, file upload)
├── backend.py           # Core logic (Parsers, BM25 algorithm, LLM router)
├── requirements.txt     # (Optional) For standard pip installs
└── temp_uploads/        # Temporary storage for uploaded documents
```

## 🛠️ Installation & Setup

This project uses uv for insanely fast Python package management.

1. Clone the repository and navigate to the folder:

```bash
cd my_bm25_bot
```

2. Install dependencies via uv:

```bash
uv add streamlit python-dotenv pymupdf rank_bm25 langchain-groq python-docx pandas openpyxl
```

3. Set up your Environment Variables:

Create a .env file in the root directory and add your Groq API key:

```
GROQ_API_KEY=gsk_your_api_key_here
```

4. Run the application:

```bash
uv run streamlit run app.py
```
The UI will automatically open in your default web browser at http://localhost:8501.