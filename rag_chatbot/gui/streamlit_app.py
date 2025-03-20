import streamlit as st
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from services.chatbot_service import ChatbotService
from pathlib import Path


def extract_document_name(file_path):
    """Extract the document name from the file path."""
    return Path(file_path).name


def display_source_metadata(metadata):
    """Display metadata in a user-friendly format."""
    st.markdown("**Metadata:**")
    st.markdown(f"- **Author:** {metadata.get('Author', 'N/A')}")
    st.markdown(f"- **Page:** {metadata.get('page', 'N/A')} of {metadata.get('total_pages', 'N/A')}")


def main():
    st.set_page_config(page_title="Chatbot NLP & LLM - UniSA", page_icon="🤖", layout="centered")
    st.title("💬 Chatbot NLP & LLM - University of Salerno")
    st.write("Ask anything about the NLP and Large Language Models course!")

    # Initialize session state
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatbotService()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_input := st.chat_input("Ask a question about the course:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream the chatbot's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            for message in st.session_state.chatbot.stream_message(user_input):
                if message['type'] != 'tool':
                    token = message["content"]
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                if message["source_documents"]:
                    st.session_state.sources = message["source_documents"]
            response_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Display sources with icons and parsed metadata
    if st.session_state.sources:
        st.sidebar.title("📚 Sources")
        for i, doc in enumerate(st.session_state.sources):
            source_metadata = doc["source"]
            document_name = extract_document_name(source_metadata.get("source", "Unknown Document"))

            # Display source with an icon and document name
            with st.sidebar.expander(f"📄 {document_name}"):
                display_source_metadata(source_metadata)
                st.markdown("**Content:**")
                st.markdown(doc["content"])


if __name__ == "__main__":
    main()