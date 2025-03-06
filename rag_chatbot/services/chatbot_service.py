import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from core.chat_graph import ChatGraph
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

# ==========================
# ChatbotService: Exposes API for GUI
# ==========================
class ChatbotService:
    def __init__(self, config={"configurable": {"thread_id": "abc123"}}):
        """Initialize chatbot components."""
        # self.retriever = Retriever()
        self.app = ChatGraph()
        self.system_prompt = SystemMessage(
        '''You are an assistant for question-answering tasks about the Natural Language Processing and Large Language Models course of
        University of Salerno. Ask only question about this topic, use also the context (if available) retrieved from the documents to answer the question.
        If you don't know the answer, say that you don't know without answering. Don't try to answer out topic questions.
        Always retrieve information before answering if the query is about NLP and Large Language Models or the course. Don't retrieve
        only if the query is not related to the course otherwise try to retrieve information.'''
    )
        
        # When we run the application, we pass in a configuration dict that specifies a thread_id. 
        # This ID is used to distinguish conversational threads (e.g., between different users).
        self.config = config

        # Initialize conversation state with system prompt
        self.app.graph.invoke({"messages": [self.system_prompt]}, self.config)


    def send_message(self, user_input: str):
        """Send a message to the chatbot and get a response."""
        input_message = {"messages": [HumanMessage(user_input)]}

        # Process the query through the graph
        response = None
        for step in self.app.graph.stream(input_message, config=self.config, stream_mode="values"):
            response = step["messages"][-1]  # Get latest response
            response.pretty_print()

        if response:
            return response  # Return only response text for GUI
        return "I'm sorry, I couldn't process your request."
    
    # # TODO: this is not working well
    # def stream_messages(self, user_input: str):
    #     """Stream messages to the chatbot and get a response."""
    #     self.conversation_state["messages"].append({"role": "user", "content": user_input})

    #     return self.chat_graph.graph.astream(self.conversation_state, stream_mode="messages")
    
    def reset_conversation(self):
        """Reset the conversation state."""
        all_messages = self.app.graph.get_state(self.config).values["messages"]
        self.app.graph.update_state(self.config, {"messages": [RemoveMessage(id=m.id) for m in all_messages if m.id != self.system_prompt.id]})

# ==========================
# Example use
# ==========================
if __name__ == "__main__":
    chatbot = ChatbotService()

    while True:
        user_input = input("Enter a query (q to quit): ")
        if user_input.lower() == "q":
            break
        if user_input.lower() == "r":
            chatbot.reset_conversation()
            print("Conversation reset.")
            continue

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
