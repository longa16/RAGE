# RAG QA on PDF with Mistral and Streamlit

## Description
This project is a RAG (Retrieval-Augmented Generation) application that allows querying a PDF document using a Mistral model via Hugging Face. The interface is built with Streamlit for simple use: upload a PDF, ask questions, and get responses based on the document's content.

## Features
- Upload and processing of PDF files.
- Creation of a vector database with FAISS and Hugging Face embeddings.
- Querying via a RetrievalQA chain with Mistral-7B-Instruct.
- Display of responses and sources.

## Installation
1. Clone the repository:
```text
git@github.com:longa16/RAGE.git
```

2. Install the dependencies:
```text
pip install -r requirements.txt
```
3. Configure your Hugging Face token in a `.env` file:

## Usage
Launch the application:
```text
streamlit run app.py
```
- Open your browser at `http://localhost:8501`.
- Upload a PDF.
- Ask a question and click "Answer".