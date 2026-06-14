from pypdf import PdfReader
from langchain_core.documents import Document


def extract_documents_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    documents = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()

        if page_text:
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": pdf_file.name,
                        "page": page_num + 1
                    }
                )
            )

    return documents