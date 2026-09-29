\# Agentic AI RAG Chatbot



\## Overview



This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions using the provided Agentic AI ebook.



\## Technologies



\- Python

\- LangChain

\- LangGraph

\- Pinecone

\- Hugging Face Embeddings

\- Ollama

\- Llama 3.2 3B

\- Streamlit



\## Features



\- Loads and processes the Agentic AI PDF

\- Splits the document into text chunks

\- Creates embeddings for the document

\- Stores embeddings in Pinecone

\- Retrieves relevant information for user questions

\- Generates answers using Llama 3.2 3B

\- Provides a Streamlit web interface

\- Refuses questions when the information is not available in the document



\## How to Run



1\. Activate the virtual environment.

2\. Start the Streamlit application using:



streamlit run app.py



3\. Open the application in the browser.

4\. Enter a question and click the Ask button.



\## Example Questions



\- What is Agentic AI?

\- What are the characteristics of Agentic AI?

\- How does Agentic AI make decisions?

\- What are the benefits of Agentic AI?



\## Project Structure



rag-agentic-ai/

├── data/

│   └── Ebook-Agentic-AI.pdf

├── src/

│   ├── \_\_init\_\_.py

│   ├── config.py

│   ├── ingestion.py

│   ├── pinecone\_setup.py

│   └── graph.py

├── app.py

├── chatbot.py

├── README.md

├── requirements.txt

└── .gitignore

