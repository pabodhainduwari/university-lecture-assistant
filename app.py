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
    background: linear-gradient(135deg, #0F172A 0%, #111827 100%);
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1E293B;
}

.main-title {
    font-size: 46px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 6px;
}

.subtitle {
    font-size: 17px;
    color: #94A3B8;
    margin-bottom: 28px;
}

.hero-card {
    padding: 30px;
    border-radius: 22px;
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    margin-bottom: 24px;
}

.info-card {
    padding: 18px;
    border-radius: 16px;
    background: #111827;
    border: 1px solid #334155;
    margin-bottom: 16px;
}

.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 14px;
    font-weight: 700;
    font-size: 15px;
    background: linear-gradient(135deg, #2563EB, #7C3AED);
    color: white;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #6D28D9);
    color: white;
    transform: scale(1.02);
}

.source-card {
    padding: 14px;
    border-radius: 12px;
    background: #020617;
    border: 1px solid #334155;
    margin-bottom: 12px;
}

.small-text {
    color: #94A3B8;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Control Panel")

    answer_language = st.selectbox(
        "Answer Language",
        ["Sinhala", "English", "Sinhala + English"]
    )

    study_mode = st.radio(
        "Study Mode",
        ["Normal", "Exam Preparation", "Quick Revision", "MCQ Practice"]
    )

    show_sources = st.checkbox(
        "Show Retrieved Sources",
        value=True
    )

    top_k = st.slider("Top Chunks", 3, 10, 5)
    temperature = st.slider("Creativity Level", 0.0, 1.0, 0.2)

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
        AI-powered study companion for lecture notes, exam preparation, summaries, MCQs and source-based answers.
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
                st.error(
                    "No readable text found. Please upload a text-based PDF, not a scanned/image PDF."
                )
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
    col1, col2, col3 = st.columns(3)

    col1.metric("PDFs Processed", st.session_state.get("pdf_count", 0))
    col2.metric("Knowledge Chunks", st.session_state.get("chunk_count", 0))
    col3.metric("Questions Asked", st.session_state.get("question_count", 0))

    st.markdown("## Ask Questions")
    st.markdown("### Quick Study Tools")

    col1, col2, col3, col4 = st.columns(4)

    quick_question = None

    with col1:
        if st.button("Summarize Lecture"):
            quick_question = "Summarize this lecture in simple points"

    with col2:
        if st.button("Exam Focus Points"):
            quick_question = "Give the most important exam points from this lecture"

    with col3:
        if st.button("Generate MCQs"):
            quick_question = "Generate 5 MCQ questions with answers from this lecture"

    with col4:
        if st.button("Simple Explanation"):
            quick_question = "Explain the main topic of this lecture in simple words"

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        if st.button("Flashcards"):
            quick_question = "Generate 10 flashcards from this lecture with question and answer format"

    with col6:
        if st.button("Definitions"):
            quick_question = "List the most important definitions from this lecture"

    with col7:
        if st.button("Short Notes"):
            quick_question = "Create short study notes from this lecture"

    with col8:
        if st.button("Essay Questions"):
            quick_question = "Generate possible exam essay questions from this lecture"

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    typed_question = st.chat_input("Ask a question from your lecture notes")
    question = quick_question or typed_question

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        st.session_state.question_count += 1

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Searching lecture notes and generating answer..."):
            answer, source_docs = get_answer_from_gemini(
                vectorstore=st.session_state.vectorstore,
                question=question,
                answer_language=answer_language,
                study_mode=study_mode,
                top_k=top_k,
                temperature=temperature
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)

        st.download_button(
            label="Download Answer",
            data=answer,
            file_name="lecture_answer.txt",
            mime="text/plain"
        )

        if show_sources and source_docs:
            with st.expander("View Retrieved Lecture Sources"):
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

                    st.text_area(
                        f"Relevant Text {i + 1}",
                        doc.page_content,
                        height=140
                    )

else:
    st.warning("Please upload and process lecture PDFs first.")