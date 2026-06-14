from langchain_google_genai import ChatGoogleGenerativeAI


def get_answer_from_gemini(vectorstore, question, answer_language):

    try:
        docs = vectorstore.similarity_search(
            question,
            k=5
        )

        context_parts = []

        for doc in docs:
            source = doc.metadata.get(
                "source",
                "Unknown PDF"
            )

            page = doc.metadata.get(
                "page",
                "Unknown page"
            )

            context_parts.append(
                f"[Source: {source}, Page: {page}]\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are a helpful university lecture assistant.

Use ONLY the lecture context below to answer the student's question.

Do not guess.

If the answer is not available in the context, say:

"I cannot find this in the uploaded lecture notes."

Answer language: {answer_language}

Context:
{context}

Question:
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

        if (
            "RESOURCE_EXHAUSTED" in error_text
            or
            "429" in error_text
        ):

            return (
                "⚠️ Gemini API quota limit reached. Please try again later.",
                []
            )

        return (
            f"⚠️ Error: {error_text}",
            []
        )