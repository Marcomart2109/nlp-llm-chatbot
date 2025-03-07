# 🤖 RAG Chatbot Test

## 🚀 Overview
This repository contains a chatbot designed for question-answering tasks about the Natural Language Processing and Large Language Models course at the University of Salerno. The chatbot uses a retrieval-augmented generation (RAG) approach to provide accurate and contextually relevant answers.

## 🛠️ Features

- **Question Answering**: The chatbot answers questions related to the NLP and Large Language Models course.
- **Contextual Retrieval**: Retrieves relevant documents from a vector store to provide contextually accurate answers.
- **Conversation Management**: Maintains conversation state and handles user interactions.

## 📦 Installation

1. Clone the repository:
    ```sh
    git clone git@github.com:Marcomart2109/nlp-llm-chatbot.git
    cd rag_chatbot_test
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add the necessary environment variables as specified in `rag_chatbot/config.py`.

## 🚀 Usage (*for now*)

1. Run the chatbot service:
    ```sh
    python rag_chatbot/services/chatbot_service.py
    ```

2. Interact with the chatbot through the command line:
    - Enter your query when prompted.
    - Type `q` to quit the interaction.

## ⚙️ Configuration

The configuration settings for the chatbot are located in `rag_chatbot/config.py`. You can adjust the following parameters:

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `DOCUMENTS_PATH`
- `VECTOR_STORE_PATH`
- `LLM_MODEL_NAME`
- `LLM_MODEL_PROVIDER`
- `EMBEDDING_MODEL`
