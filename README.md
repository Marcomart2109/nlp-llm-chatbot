# 🤖 RAG Chatbot Test

## 🚀 Overview
This repository contains a chatbot designed for question-answering tasks about the Natural Language Processing and Large Language Models course at the University of Salerno. The chatbot uses a retrieval-augmented generation (RAG) approach to provide accurate and contextually relevant answers.

## 🛠️ Features

- **Question Answering**: The chatbot answers questions related to the NLP and Large Language Models course.
- **Contextual Retrieval**: Retrieves relevant documents from a vector store to provide contextually accurate answers.
- **Conversation Management**: Maintains conversation state and handles user interactions.
- **Interactive GUI**: A Streamlit-based interface for easy interaction with real-time response streaming.
- **Source Documentation**: View the sources used to generate answers, with expandable details.

## 📦 Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Marcomart2109/nlp-llm-chatbot.git
    cd nlp-llm-chatbot
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

5. **Important**: Make sure to agree to the terms of the gated repository `mistralai/Mixtral-8x7B-v0.1` on [this link](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) so the model in the code can download its tokenizer from the hub.

## 🚀 Usage

### Streamlit GUI

1. Start the Streamlit application:
    ```sh
    streamlit run rag_chatbot/gui/streamlit_app.py
    ```

2. The Streamlit interface will open in your default web browser at `http://localhost:8501`.

3. You can:
   - Type questions in the input box and press Enter or click "Send"
   - View the chatbot's responses in real-time as they stream in
   - Explore the source documents used to generate answers in the sidebar
   - Reset the conversation when needed

### Command Line Interface

Alternatively, you can run the chatbot via command line:

1. Run the chatbot service:
    ```sh
    python rag_chatbot/services/chatbot_service.py
    ```

2. Interact with the chatbot through the command line:
    - Enter your query when prompted.
    - Type `q` to quit the interaction.
    - Type `r` to reset the conversation.

## ⚙️ Configuration

The configuration settings for the chatbot are located in `rag_chatbot/config.py`. You can adjust the following parameters:

- `CHUNK_SIZE`: Size of text chunks for embedding
- `CHUNK_OVERLAP`: Overlap between chunks to maintain context
- `TOP_K`: Number of documents to retrieve
- `DOCUMENTS_PATH`: Path to the course documents
- `VECTOR_STORE_PATH`: Path to store the vector embeddings
- `LLM_MODEL_NAME`: Name of the language model to use
- `LLM_MODEL_PROVIDER`: Provider of the language model
- `MISTRAL_EMBEDDING_MODEL`: Embedding model for Mistral AI
- `HF_EMBEDDING_MODEL`: Embedding model for HuggingFace
- `TEMPERATURE`: Temperature parameter for response generation

## 📚 Project Structure

```
rag_chatbot_test/
├── data/
│   ├── slides_and_syllabus/  # Course documents
│   └── vector_store/         # Generated vector embeddings
├── rag_chatbot/
│   ├── core/
│   │   ├── chat_graph.py     # Conversation flow definition
│   │   └── vector_store.py   # Vector store management
│   ├── gui/
│   │   └── streamlit_app.py  # Streamlit GUI application
│   └── services/
│       └── chatbot_service.py # Main chatbot service
├── .env                      # Environment variables (not in repo)
├── config.py                 # Configuration settings
└── requirements.txt          # Project dependencies
```
