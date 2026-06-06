import os
import shutil
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
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    return vectorstore


def get_answer_from_gemini(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=5)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful university lecture assistant.

Use the lecture context below to answer the question.
If the exact answer is not directly stated but can be explained from the context, explain it clearly.
If the answer is completely unrelated to the context, say:
"I cannot find this in the uploaded lecture notes."

Answer in simple student-friendly language.

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


st.set_page_config(
    page_title="University Lecture Assistant",
    page_icon="📚"
)

st.title("📚 University Lecture Assistant")
st.write("Upload your lecture PDFs, create a vector database, and ask questions.")

uploaded_files = st.file_uploader(
    "Upload Lecture PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""

    for pdf in uploaded_files:
        st.write(f"📄 {pdf.name}")
        all_text += extract_text_from_pdf(pdf)

    st.subheader("Extracted Text Preview")
    st.text_area("Preview", all_text[:3000], height=300)

    chunks = split_text_into_chunks(all_text)

    st.subheader("Chunk Information")
    st.write(f"Total Chunks Created: {len(chunks)}")

    if chunks:
        st.text_area("First Chunk", chunks[0], height=250)

    if st.button("Create Vector Database"):
        with st.spinner("Creating embeddings and saving to ChromaDB..."):
            vectorstore = create_vector_store(chunks)
            st.session_state.vectorstore = vectorstore

        st.success("Embeddings created and saved to ChromaDB successfully!")


if "vectorstore" in st.session_state:
    st.subheader("Ask Questions")

    question = st.text_input("Ask a question from your lecture notes")

    if question:
        with st.spinner("Searching lecture notes and generating answer..."):
            answer, source_docs = get_answer_from_gemini(
                st.session_state.vectorstore,
                question
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Lecture Parts")
        for i, doc in enumerate(source_docs):
            st.text_area(
                f"Source {i + 1}",
                doc.page_content,
                height=180
            )