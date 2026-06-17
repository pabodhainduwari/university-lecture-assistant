# StudyMate-AI-University Lecture Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system that enables students to upload lecture PDFs and interact with their study materials through natural language questions.

Built using Python, Streamlit, LangChain, ChromaDB, and Google Gemini, the system transforms lecture notes into an intelligent knowledge base for learning, revision, and exam preparation.

---

## Features

### Document Processing

* Upload multiple lecture PDFs
* Automatic text extraction from PDF documents
* Intelligent document chunking
* Vector database creation using ChromaDB

### AI-Powered Question Answering

* Ask questions directly from lecture notes
* Context-aware responses using Retrieval-Augmented Generation (RAG)
* Source-based answer generation
* Multi-language support:

  * English
  * Sinhala
  * Sinhala + English

### Study Tools

* Lecture Summaries
* Exam Focus Points
* MCQ Generation
* Flashcard Generation
* Definitions Extraction
* Short Notes Creation
* Essay Question Generation
* Simple Topic Explanations

### User Experience

* Modern Streamlit Interface
* Multi-PDF Support
* Session-Based Chat History
* Download Generated Answers
* Source Reference Display
* Study Mode Selection

---

## System Architecture

PDF Upload
↓
Text Extraction
↓
Document Chunking
↓
Embedding Generation
↓
Chroma Vector Database
↓
Similarity Search
↓
Gemini LLM
↓
Answer Generation

---

## Technologies Used

### Programming Language

* Python

### Frontend

* Streamlit

### AI & LLM

* Google Gemini API
* LangChain

### Vector Database

* ChromaDB

### NLP & Document Processing

* PyPDF
* LangChain Text Splitters

### Environment Management

* Python Dotenv

### Version Control

* Git
* GitHub

---

## Project Structure

```text
University-Lecture-Assistant/
│
├── app.py
│
├── utils/
│   ├── pdf_processor.py
│   ├── chunking.py
│   ├── vector_store.py
│   └── llm.py
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/pabodhainduwari/university-lecture-assistant.git
cd university-lecture-assistant
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:


Get your Gemini API key from:

https://aistudio.google.com/apikey

---

## Run Application

```bash
streamlit run app.py
```

---

## How It Works

1. Upload lecture PDFs.
2. Extract text from documents.
3. Split content into manageable chunks.
4. Generate embeddings using Gemini Embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks for user questions.
7. Generate context-aware answers using Gemini.
8. Display responses with supporting sources.

---

## Future Improvements

* OCR Support for Scanned PDFs
* Chat Export to PDF
* Conversation Memory
* Lecture Summarization Dashboard
* Quiz Generation with Difficulty Levels
* Voice-Based Question Answering
* Multi-Model Support (Gemini, OpenAI, Claude)
* Cloud Deployment Support

---

## Author

Pabodha Marasinghe

Data Science Undergraduate
Sri Lanka Institute of Information Technology (SLIIT)

GitHub: https://github.com/pabodhainduwari

---

## License

This project is developed for educational and portfolio purposes.
