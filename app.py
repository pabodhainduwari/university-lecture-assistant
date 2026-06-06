import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def split_text_into_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    return chunks


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()

        if page_text:
            text += f"\n\n--- Page {page_num + 1} ---\n"
            text += page_text

    return text


st.set_page_config(
    page_title="University Lecture Assistant",
    page_icon="📚"
)

st.title("📚 University Lecture Assistant")

uploaded_files = st.file_uploader(
    "Upload Lecture PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_text = ""

    for pdf in uploaded_files:
        all_text += extract_text_from_pdf(pdf)

    st.subheader("Extracted Text Preview")

    st.text_area(
        "Preview",
        all_text[:3000],
        height=300
    )

    chunks = split_text_into_chunks(all_text)

    st.subheader("Chunk Information")

    st.write(f"Total Chunks Created: {len(chunks)}")

    if chunks:
        st.text_area(
            "First Chunk",
            chunks[0],
            height=250
        )