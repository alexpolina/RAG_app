# RAG System with buid based on Streamlit

This repository contains a simple Retrieval-Augmented Generation (RAG) application that:
1. Lets you upload a PDF file.
2. Uses API (OpenAI-compatible) for embeddings and chat completions.
3. Stores vector embeddings in FAISS.
4. Serves a Streamlit web interface for question-answering.

## Features
- PDF upload
- Chunking of PDF text
- Embedding
- Vector search using FAISS
- GPT-like AI responses (ChatCompletion)
- Streamlit UI

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/rag_app.git
   cd rag_app
