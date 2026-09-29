import streamlit as st

from src.graph import graph


st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖"
)


st.title("🤖 Agentic AI RAG Chatbot")

st.write(
    "Ask questions about the provided Agentic AI document."
)


question = st.text_input("Enter your question:")


if st.button("Ask"):

    if question.strip():

        with st.spinner(
            "Searching the document and generating an answer..."
        ):

            result = graph.invoke({
                "question": question
            })


        st.subheader("Answer")

        st.write(result["answer"])


        st.subheader("Retrieved Chunks")

        for i, document in enumerate(result["context"], 1):

            st.write(f"**Chunk {i}**")

            st.write(document.page_content)


        st.subheader("Relevance Scores")

        for i, score in enumerate(result["scores"], 1):

            st.write(
                f"Chunk {i}: {score:.4f}"
            )

    else:

        st.warning("Please enter a question.")