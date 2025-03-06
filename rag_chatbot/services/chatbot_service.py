import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from core.chat_graph import ChatGraph
from langchain_core.messages import SystemMessage

# ==========================
# ChatbotService: Exposes API for GUI
# ==========================
class ChatbotService:
    def __init__(self):
        """Initialize chatbot components."""
        # self.retriever = Retriever()
        self.chat_graph = ChatGraph()
        self.system_prompt = SystemMessage(
        '''You are an assistant for question-answering tasks about the Natural Language Processing and Large Language Models course of
        University of Salerno. Ask only question about this topic, use also the context (if available) retrieved from the documents to answer the question.
        If you don't know the answer, say that you don't know without answering. Don't try to answer out topic questions.
        Always retrieve information before answering if the query is about NLP and Large Language Models or the course. Don't retrieve
        only if the query is not related to the course otherwise try to retrieve information.'''
    )
        self.conversation_state = {"messages": [self.system_prompt]}

    def send_message(self, user_input: str):
        """Send a message to the chatbot and get a response."""
        self.conversation_state["messages"].append({"role": "user", "content": user_input})

        # Process the query through the graph
        response = None
        for step in self.chat_graph.graph.stream(self.conversation_state, stream_mode="values"):
            response = step["messages"][-1]  # Get latest response
            response.pretty_print()

        # Append response to conversation history
        if response:
            self.conversation_state["messages"].append(response)
            return response  # Return only response text for GUI
        return "I'm sorry, I couldn't process your request."
    
    # TODO: this is not working well
    def stream_messages(self, user_input: str):
        """Stream messages to the chatbot and get a response."""
        self.conversation_state["messages"].append({"role": "user", "content": user_input})

        return self.chat_graph.graph.astream(self.conversation_state, stream_mode="messages")
    
    def reset_conversation(self):
        """Reset the conversation state."""
        self.conversation_state = {"messages": [self.system_prompt]}

# ==========================
# Example use
# ==========================
if __name__ == "__main__":
    chatbot = ChatbotService()

    while True:
        user_input = input("Enter a query (q to quit): ")
        if user_input.lower() == "q":
            break

        response = chatbot.send_message(user_input)
        # response.pretty_print()


# async def main():
#     chatbot = ChatbotService()

#     while True:
#         user_input = input("Enter a query (q to quit): ")
#         if user_input.lower() == "q":
#             break

#         async for message, metadata in chatbot.stream_messages(user_input):
#             print(message.content, end="", flush=True)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
