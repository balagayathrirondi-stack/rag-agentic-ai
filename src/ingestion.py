from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PDF_PATH = "data/Ebook-Agentic-AI.pdf"


def load_and_split_pdf():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    chunks = load_and_split_pdf()
    print(f"PDF loaded successfully.")
    print(f"Total chunks: {len(chunks)}")