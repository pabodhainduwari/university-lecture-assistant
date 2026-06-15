from langchain_google_genai import ChatGoogleGenerativeAI


def get_answer_from_gemini(
    vectorstore,
    question,
    answer_language,
    study_mode="Normal"
):
    try:
        docs = vectorstore.similarity_search(question, k=5)

        context_parts = []

        for doc in docs:
            source = doc.metadata.get("source", "Unknown PDF")
            page = doc.metadata.get("page", "Unknown page")

            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a helpful university lecture assistant.

Use ONLY the lecture context below to answer the student's question.
Do not guess or add unsupported information.

If the answer is not clearly available in the context, say:
"I cannot find this in the uploaded lecture notes."

Answer language: {answer_language}
Study mode: {study_mode}

Instructions:
- Give a clear, simple, student-friendly answer.
- Use bullet points where useful.
- Do not include source names or page numbers inside the answer.
- If the study mode is Exam Preparation, focus on exam-relevant points.
- If the study mode is Quick Revision, keep the answer short and clear.
- If the study mode is MCQ Practice, generate questions with correct answers.

Lecture Context:
{context}

Student Question:
{question}

Answer:
"""

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2
        )

        response = llm.invoke(prompt)
        return response.content, docs

    except Exception as e:
        error_text = str(e)

        if "RESOURCE_EXHAUSTED" in error_text or "429" in error_text:
            return (
                "Gemini API quota limit reached. Please wait and try again later.",
                []
            )

        return (
            f"An error occurred: {error_text}",
            []
        )