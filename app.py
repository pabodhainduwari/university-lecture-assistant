import streamlit as st
from dotenv import load_dotenv

from utils.pdf_processor import extract_documents_from_pdf
from utils.chunking import split_documents_into_chunks
from utils.vector_store import create_vector_store
from utils.llm import get_answer_from_gemini

load_dotenv()

st.set_page_config(
    page_title="University Lecture Assistant",
    page_icon="ULA",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #1E3A8A 0%, #0F172A 35%, #020617 100%);
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0F172A 100%);
    border-right: 1px solid #1E293B;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC;
}

.sidebar-header {
    padding: 20px;
    border-radius: 20px;
    background: linear-gradient(135deg, #1E293B, #111827);
    border: 1px solid #334155;
    margin-bottom: 22px;
}

.sidebar-header-title {
    font-size: 22px;
    font-weight: 800;
    color: #F8FAFC;
}

.sidebar-header-subtitle {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 4px;
}

.main-title {
    font-size: 56px;
    font-weight: 900;
    background: linear-gradient(90deg, #60A5FA, #A855F7, #F472B6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 17px;
    color: #94A3B8;
    margin-bottom: 28px;
}

.hero-card {
    padding: 46px;
    border-radius: 30px;
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 18px 45px rgba(0,0,0,0.35);
    margin-bottom: 30px;
}

.user-card {
    padding: 18px;
    border-radius: 16px;
    background: rgba(30, 41, 59, 0.95);
    border-left: 5px solid #3B82F6;
    margin-bottom: 16px;
}

.assistant-card {
    padding: 18px;
    border-radius: 16px;
    background: rgba(17, 24, 39, 0.95);
    border-left: 5px solid #8B5CF6;
    margin-bottom: 20px;
}

.stat-card {
    background: rgba(15, 23, 42, 0.92);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid #334155;
    text-align: center;
    margin-bottom: 18px;
}

.stat-card h4 {
    color: #94A3B8;
    margin-bottom: 8px;
    font-size: 14px;
}

.stat-card h2 {
    color: #F8FAFC;
    font-size: 34px;
    margin: 0;
}

.stButton > button {
    width: 100%;
    height: 62px;
    border-radius: 18px;
    font-weight: 800;
    font-size: 15px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    color: white;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.22);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #6D28D9);
    color: white;
    transform: translateY(-2px);
    border: 1px solid rgba(255,255,255,0.25);
}

.quick-title {
    font-size: 32px;
    font-weight: 900;
    color: #F8FAFC;
    margin-top: 35px;
    margin-bottom: 6px;
}

.quick-subtitle {
    font-size: 15px;
    color: #94A3B8;
    margin-bottom: 22px;
}

.source-card {
    padding: 14px;
    border-radius: 12px;
    background: #020617;
    border: 1px solid #334155;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-header-title">Study Control</div>
        <div class="sidebar-header-subtitle">Configure your learning session</div>
    </div>
    """, unsafe_allow_html=True)

    answer_language = st.selectbox(
        "Answer Language",
        ["Sinhala", "English", "Sinhala + English"]
    )

    study_mode = st.radio(
        "Study Mode",
        ["Normal", "Exam Preparation", "Quick Revision", "MCQ Practice"]
    )

    st.divider()

    st.markdown("### System Status")

    if "vectorstore" in st.session_state:
        st.success("Knowledge Base Ready")
    else:
        st.warning("No Lecture Processed")

    st.metric("PDFs Loaded", st.session_state.get("pdf_count", 0))
    st.metric("Knowledge Chunks", st.session_state.get("chunk_count", 0))
    st.metric("Questions Asked", st.session_state.get("question_count", 0))

    st.divider()

    if "uploaded_pdf_names" in st.session_state:
        st.markdown("### Uploaded Lectures")
        for name in st.session_state.uploaded_pdf_names:
            st.write(f"- {name}")

    st.divider()

    if st.button("Clear Current Lecture Data"):
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("messages", None)
        st.session_state.pop("pdf_count", None)
        st.session_state.pop("chunk_count", None)
        st.session_state.pop("question_count", None)
        st.session_state.pop("uploaded_pdf_names", None)
        st.success("Lecture data cleared.")

st.markdown("""
<div class="hero-card">
    <div class="main-title">University Lecture Assistant</div>
    <div class="subtitle">
        AI-powered study companion for lecture notes, summaries, MCQs, short notes and exam preparation.
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Lecture PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} PDF file(s) selected.")

    with st.expander("View selected files"):
        for pdf in uploaded_files:
            st.write(f"- {pdf.name}")

    if st.button("Process Lecture PDFs"):
        with st.spinner("Reading PDFs and building knowledge base..."):
            all_documents = []

            for pdf in uploaded_files:
                all_documents.extend(extract_documents_from_pdf(pdf))

            if not all_documents:
                st.error("No readable text found. Please upload a text-based PDF, not a scanned/image PDF.")
                st.stop()

            chunks = split_documents_into_chunks(all_documents)

            if not chunks:
                st.error("No text chunks were created. Please try another PDF.")
                st.stop()

            vectorstore = create_vector_store(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []
            st.session_state.pdf_count = len(uploaded_files)
            st.session_state.chunk_count = len(chunks)
            st.session_state.question_count = 0
            st.session_state.uploaded_pdf_names = [pdf.name for pdf in uploaded_files]

        st.success("Lecture PDFs processed successfully. You can now ask questions.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

if "vectorstore" in st.session_state:
    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            f"""
            <div class="stat-card">
                <h4>PDFs Processed</h4>
                <h2>{st.session_state.get("pdf_count", 0)}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with stat2:
        st.markdown(
            f"""
            <div class="stat-card">
                <h4>Knowledge Chunks</h4>
                <h2>{st.session_state.get("chunk_count", 0)}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with stat3:
        st.markdown(
            f"""
            <div class="stat-card">
                <h4>Questions Asked</h4>
                <h2>{st.session_state.get("question_count", 0)}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="quick-title">Study Tools</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="quick-subtitle">Choose a study action or ask your own question below.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    quick_question = None

    with col1:
        if st.button("📌 Summarize Lecture"):
            quick_question = "Summarize this lecture in simple points"

    with col2:
        if st.button("🎯 Exam Focus Points"):
            quick_question = "Give the most important exam points from this lecture"

    with col3:
        if st.button("📝 Generate MCQs"):
            quick_question = "Generate 5 MCQ questions with answers from this lecture"

    with col4:
        if st.button("💡 Simple Explanation"):
            quick_question = "Explain the main topic of this lecture in simple words"

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if st.button("🃏 Flashcards"):
            quick_question = "Generate 10 flashcards from this lecture with question and answer format"

    with col6:
        if st.button("📖 Definitions"):
            quick_question = "List the most important definitions from this lecture"

    with col7:
        if st.button("🗒️ Short Notes"):
            quick_question = "Create short study notes from this lecture"

    with col8:
        if st.button("✍️ Essay Questions"):
            quick_question = "Generate possible exam essay questions from this lecture"

    st.divider()

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="user-card"><b>You</b><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="assistant-card"><b>Assistant</b><br>{message["content"]}</div>',
                unsafe_allow_html=True
            )

    typed_question = st.chat_input("Ask a question from your lecture notes")
    question = quick_question or typed_question

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        st.session_state.question_count += 1

        st.markdown(
            f'<div class="user-card"><b>You</b><br>{question}</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Searching lecture notes and generating answer..."):
            answer, source_docs = get_answer_from_gemini(
                vectorstore=st.session_state.vectorstore,
                question=question,
                answer_language=answer_language,
                study_mode=study_mode
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        st.markdown(
            f'<div class="assistant-card"><b>Assistant</b><br>{answer}</div>',
            unsafe_allow_html=True
        )

        st.download_button(
            label="Download Answer",
            data=answer,
            file_name="lecture_answer.txt",
            mime="text/plain"
        )

        if source_docs:
            with st.expander("View Sources"):
                for i, doc in enumerate(source_docs):
                    source = doc.metadata.get("source", "Unknown PDF")
                    page = doc.metadata.get("page", "Unknown page")

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>Source {i + 1}</b><br>
                            File: {source}<br>
                            Page: {page}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

else:
    st.warning("Please upload and process lecture PDFs first.")