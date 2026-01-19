
# doc_read.py
import io
import os
import hashlib
import streamlit as st

from PyPDF2 import PdfReader

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    ChatHuggingFace,
    HuggingFaceEndpoint,
)

# =========================
# BASIC APP CONFIG
# =========================
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="centered")
st.title("PDF‑Powered RAG Chatbot (Hugging Face + FAISS + Memory)")
st.caption("Upload a PDF → we chunk + embed it → retrieve relevant context → answer with a Hugging Face LLM.")

# Helpful hint if launched incorrectly
if st.runtime.exists() is False:  # requires Streamlit >= 1.32; if older, this line is simply ignored
    st.warning("Run this app with:  \n`streamlit run doc_read.py`")
    # Do not st.stop() here; let the rest run for older versions gracefully.

# =========================
# HUGGING FACE TOKEN
# =========================
hf_token = st.text_input("Enter your Hugging Face API Token", type="password", help="Create one at https://huggingface.co/settings/tokens")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

if not hf_token:
    st.warning("Please enter your Hugging Face token to continue.")
    st.stop()

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("📤 Upload a PDF document", type=["pdf"])
if uploaded_file is None:
    st.info("Upload a PDF to proceed.")
    st.stop()

# =========================
# PDF TEXT EXTRACTION (robust)
# =========================
def extract_pdf_text(file_obj) -> str:
    """
    Convert Streamlit UploadedFile -> BytesIO -> PdfReader and extract text safely.
    Handles encrypted PDFs (best-effort), empty pages, and missing text.
    """
    try:
        pdf_bytes = file_obj.getvalue()
        if not pdf_bytes:
            return ""

        reader = PdfReader(io.BytesIO(pdf_bytes))

        # Best-effort decrypt (if applicable)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # attempt empty password
            except Exception:
                return ""

        contents = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
                contents.append(txt)
            except Exception:
                # Skip problematic page, continue
                continue

        return "\n".join(contents).strip()
    except Exception:
        return ""

with st.spinner("Extracting text from PDF..."):
    pdf_text = extract_pdf_text(uploaded_file)

if not pdf_text:
    st.error("No extractable text found. The PDF may be scanned or encrypted. Try a different PDF or enable OCR.")
    st.stop()

st.success("✅ PDF text extracted.")
st.write(f"📄 Characters extracted: {len(pdf_text):,}")

# =========================
# CHUNKING
# =========================
st.sidebar.header("Chunking & Model Settings")
chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=2000, value=800, step=50)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=500, value=150, step=10)

splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
raw_chunks = splitter.split_text(pdf_text)

# Filter out empty/whitespace chunks to prevent FAISS errors
chunks = [c.strip() for c in raw_chunks if c and c.strip()]

if not chunks:
    st.error("No valid chunks produced from the PDF text. Try adjusting chunk size/overlap or use a different PDF.")
    st.stop()

st.write(f"🧩 Chunks created: {len(chunks):,}")

# =========================
# CACHE KEY
# =========================
def index_key_from_content(text: str, model_name: str, csize: int, coverlap: int) -> str:
    m = hashlib.md5()
    # To avoid extremely large caching keys, hash only the first N chars if needed:
    sample = text if len(text) < 2_000_000 else text[:2_000_000]
    m.update(sample.encode("utf-8"))
    m.update(model_name.encode("utf-8"))
    m.update(f"{csize}-{coverlap}".encode("utf-8"))
    return m.hexdigest()

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
cache_key = index_key_from_content(pdf_text, EMBED_MODEL, chunk_size, chunk_overlap)

# =========================
# BUILD INDEX (cached)
# Avoid spinner inside cache to prevent NoSessionContext in odd environments
# =========================
@st.cache_resource(show_spinner=False)
def build_index(_cache_key: str, _chunks: list):
    if not _chunks:
        raise ValueError("No chunks to index.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_texts(_chunks, embeddings)
    return db.as_retriever(search_kwargs={"k": 4})

with st.spinner("Building vector index..."):
    retriever = build_index(cache_key, chunks)

# =========================
# HUGGING FACE CHAT MODEL
# =========================
model_id = st.sidebar.selectbox(
    "Hugging Face model",
    options=[
        "mistralai/Mistral-7B-Instruct-v0.2",
        "HuggingFaceH4/zephyr-7b-beta",
        "tiiuae/falcon-7b-instruct",
        "openai/gpt-oss-20b",
    ],
    index=0,
)

max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 256, 32)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
repetition_penalty = st.sidebar.slider("Repetition penalty", 1.0, 2.0, 1.05, 0.01)

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        timeout=120,
    )
)

# =========================
# HISTORY STORE (multi-session)
# =========================
if "store" not in st.session_state:
    st.session_state.store = {}

def get_history(session_id: str) -> BaseChatMessageHistory:
    store = st.session_state.store
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# =========================
# PROMPT
# =========================
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Use the retrieved document context to answer the user. "
     "If the answer cannot be found in the context, state that briefly, then answer from general knowledge."),
    MessagesPlaceholder("history"),
    ("human", "{question}\n\nContext:\n{context}")
])

# =========================
# RAG PIPELINE (retrieve -> prompt -> llm) with managed history
# =========================
def build_context(input_dict):
    question = input_dict["question"]
    docs = retriever.invoke(question)
    return "\n".join([d.page_content for d in docs])

runnable = RunnablePassthrough.assign(context=build_context)
rag_chain = runnable | prompt | llm

chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)

# =========================
# UI: CHAT
# =========================
st.divider()
session_id = st.text_input("💬 Session ID (use different IDs for separate conversations)", value="session1")
user_question = st.text_input("Ask a question about this PDF")

col1, col2 = st.columns([1, 1])
with col1:
    send_clicked = st.button("Send", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear Session History", use_container_width=True)

if clear_clicked:
    if session_id in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    st.success(f"Cleared history for '{session_id}'.")

if send_clicked:
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            response = chat_chain.invoke(
                {"question": user_question},
                config={"configurable": {"session_id": session_id}}
            )
        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(response.content)

# Show conversation history
if session_id in st.session_state.store and st.session_state.store[session_id].messages:
    st.subheader("📜 Conversation History")
    for msg in st.session_state.store[session_id].messages:
        role = "user" if msg.type == "human" else "assistant"
        st.chat_message(role).write(msg.content)

