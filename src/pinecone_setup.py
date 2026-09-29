from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.ingestion import load_and_split_pdf
from src.config import PINECONE_INDEX_NAME


def upload_to_pinecone():
    chunks = load_and_split_pdf()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print(f"Successfully uploaded {len(chunks)} chunks to Pinecone.")


if __name__ == "__main__":
    upload_to_pinecone()