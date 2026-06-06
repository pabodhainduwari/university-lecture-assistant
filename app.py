import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

load_dotenv()


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n\n--- Page {page_num + 1} ---\n"
            text += page_text

    return text


def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    return text_splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vectorstore


def get_answer_from_gemini(vectorstore, question, answer_language):
    try:
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful university lecture assistant.

Use the lecture context below to answer the student's question.
If the answer is not available in the context, say:
"I cannot find this in the uploaded lecture notes."

Answer language: {answer_language}

Give a clear, simple, student-friendly answer.

Context:
{context}

Question:
{question}

Answer:
"""

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

        response = llm.invoke(prompt)
        return response.content, docs

    except Exception as e:
        error_text = str(e)

        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            return (
                "⚠️ Gemini API quota limit reached. Please wait 5–10 minutes and try again.",
                []
            )

        return (
            f"⚠️ An error occurred: {error_text}",
            []
        )


st.set_page_config(
    page_title="University Lecture Assistant",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 48px;
    font-weight: 800;
    margin-bottom: 5px;
}
.subtitle {
    font-size: 18px;
    color: #b8b8b8;
    margin-bottom: 25px;
}
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 14px;
    font-weight: 700;
    font-size: 16px;
    border: 1px solid #4b5563;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white;
}
.stButton > button:hover {
    border: 1px solid #ffffff;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">📚 University Lecture Assistant</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Chat with your lecture PDFs instantly. Upload → Process → Ask → Learn.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ Settings")

    answer_language = st.selectbox(
        "Answer language",
        ["Sinhala", "English", "Sinhala + English"]
    )

    show_sources = st.checkbox(
        "Show retrieved lecture sources",
        value=True
    )

    if st.button("🧹 Clear Current Lecture Data"):
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("messages", None)
        st.success("Lecture data cleared.")


uploaded_files = st.file_uploader(
    "📤 Upload Lecture PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"✅ {len(uploaded_files)} PDF file(s) selected.")

    if st.button("🚀 Process Lecture PDFs"):
        with st.spinner("Reading PDFs and preparing lecture knowledge base..."):
            all_text = ""

            for pdf in uploaded_files:
                all_text += extract_text_from_pdf(pdf)

            chunks = split_text_into_chunks(all_text)
            vectorstore = create_vector_store(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []

        st.success("✅ Lecture PDFs processed successfully. You can now ask questions.")


if "messages" not in st.session_state:
    st.session_state.messages = []


if "vectorstore" in st.session_state:
    st.markdown("## 💬 Ask Questions")
    st.markdown("### ⚡ Quick Study Tools")

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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    typed_question = st.chat_input("Ask another question from your lecture notes...")
    question = quick_question or typed_question

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Searching your lecture notes and generating answer..."):
            answer, source_docs = get_answer_from_gemini(
                st.session_state.vectorstore,
                question,
                answer_language
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)

        if show_sources and source_docs:
            with st.expander("📌 View retrieved lecture sources"):
                for i, doc in enumerate(source_docs):
                    st.text_area(
                        f"Source {i + 1}",
                        doc.page_content,
                        height=160
                    )

else:
    st.warning("Please upload and process lecture PDFs first.")