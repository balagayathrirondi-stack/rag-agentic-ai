from typing import TypedDict

from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

from src.config import PINECONE_INDEX_NAME


# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)


# RAG State
class RAGState(TypedDict):
    question: str
    context: list
    answer: str
    scores: list


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Pinecone Vector Store
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)


# Prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question ONLY using the context below.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)


# Retrieve relevant chunks
def retrieve(state: RAGState):
    results = vectorstore.similarity_search_with_score(
        state["question"],
        k=6
    )

    documents = [item[0] for item in results]
    scores = [item[1] for item in results]

    return {
        "context": documents,
        "scores": scores
    }


# Generate answer
def generate(state: RAGState):
    context_text = "\n\n".join(
        document.page_content
        for document in state["context"]
    )

    messages = prompt.format_messages(
        context=context_text,
        question=state["question"]
    )

    response = llm.invoke(messages)

    answer = response.content

    # Gemini may return structured content.
    # Extract only the actual text.
    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return {
        "answer": answer
    }


# Build LangGraph workflow
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()