import streamlit as st
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from services.chatbot_service import ChatbotService

# Streamlit UI setup
st.set_page_config(page_title="Chatbot NLP & LLM - UniSA", page_icon="🤖", layout="centered")
st.title("💬 Chatbot NLP & LLM - University of Salerno")
st.write("Ask anything about the NLP and Large Language Models course!")

# Apply custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #6200EE;
            color: white;
            border-radius: 12px;
            padding: 12px 20px;
            font-size: 16px;
        }
        .stTextInput>div>input {
            background-color: #2C2C2C;
            color: white;
            border-radius: 8px;
            border: 1px solid #6200EE;
        }
        .stForm {
            background-color: #1A1A1A;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stChatMessage {
            background-color: #333333;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .stChatMessage>div {
            color: white;
        }
        .stSpinner>div {
            color: #6200EE;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize chatbot service if not already done
if "chatbot_service" not in st.session_state:
    st.session_state["chatbot_service"] = ChatbotService()
    st.session_state["messages"] = []

def send_message(user_input):
    """Process user input and generate a chatbot response."""
    chatbot_service = st.session_state["chatbot_service"]
    bot_response = chatbot_service.send_message(user_input)
    return bot_response.content if bot_response else "I'm sorry, I can't process your request."

# Utilizza un form per inviare il messaggio con Invio
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Write your question here:", key="user_input", max_chars=300)
    submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        # Aggiungi la domanda dell'utente immediatamente alla conversazione
        st.session_state["messages"].append(("🧑‍🎓", user_input))
        
        # Mostra la risposta di "Processing" mentre il bot elabora
        st.session_state["messages"].append(("🤖", "Processing..."))
        st.session_state["messages"] = st.session_state["messages"]  # Trigger per aggiornare la UI

        # Crea un placeholder per la risposta animata
        placeholder = st.empty()

        with st.spinner("Processing your request..."):
            # Ottieni la risposta del bot
            bot_response = send_message(user_input)
        
        words = bot_response.split()
        animated_response = ""
        for word in words:
            animated_response += word + " "
            placeholder.markdown(f"**🤖:** {animated_response}")
            time.sleep(0.1) 

        # Sostituisci "Processing..." con la risposta finale
        st.session_state["messages"][-1] = ("🤖", bot_response)
        
        st.rerun()

# Display conversation
for sender, message in reversed(st.session_state["messages"]):
    with st.chat_message(sender):
        st.write(message)

# Reset conversation
if st.button("🔄 Reset Conversation", use_container_width=True):
    chatbot_service = st.session_state["chatbot_service"]
    chatbot_service.reset_conversation()
    st.session_state["messages"] = []
    st.rerun()
