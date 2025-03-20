import sys
import os
import uuid
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.chat_graph import ChatGraph
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage

# ==========================
# ChatbotService: Exposes API for GUI
# ==========================
class ChatbotService:
    def __init__(self):
        """Initialize chatbot components."""
        # self.retriever = Retriever()
        self.app = ChatGraph()
        self.system_prompt = SystemMessage(
        '''You are an assistant for question-answering tasks about the Natural Language Processing and Large Language Models course of
        University of Salerno. Ask only question about this topic, use also the context (if available) retrieved from the documents to answer the question.
        You must ONLY answer questions related to the course, recognizing out-of-context questions and responding with "I'm sorry, I'm not enabled to provide answers
        on topics outside of the course."
        Always retrieve information before answering if the query is about NLP and Large Language Models or the course. Don't retrieve
        only if the query is not related to the course otherwise try to retrieve information.'''
    )
        self.thread_id = self._generate_thread_id()
        
        # When we run the application, we pass in a configuration dict that specifies a thread_id. 
        # This ID is used to distinguish conversational threads (e.g., between different users).
        self.config = {"configurable": {"thread_id": self.thread_id}}

        # Initialize conversation state with system prompt
        self.app.graph.invoke({"messages": [self.system_prompt]}, self.config)
    
    def _generate_thread_id(self):
        """Generate a unique thread ID for each session."""
        return str(uuid.uuid4())


    def send_message(self, user_input: str, verbose=False):
        """Send a message to the chatbot and get a response."""
        input_message = {"messages": [HumanMessage(user_input)]}
        
        # Initialize variables to store the response and source documents
        response = None
        source_documents = []
        
        # Process the query through the graph
        for step in self.app.graph.stream(input_message, config=self.config, stream_mode="values"):
            response = step["messages"][-1]  # Get latest response
            
            # Check if this is a tool message (retrieval result)
            if response.type == "tool":
                parsed_results = self._parse_retrieved_text(response.content)
                source_documents = parsed_results

            if verbose:
                response.pretty_print()

        if response:
            # Return dictionary with response content, source documents, and message type
            return {
                "content": response.content,
                "source_documents": source_documents,
                "type": response.type
            }
        return {"content": "I'm sorry, I couldn't process your request.", "source_documents": [], "type": "ai"}

    
    def stream_message(self, user_input: str):
        """Stream the chatbot's response token by token."""
        input_message = {"messages": [HumanMessage(user_input)]}
        
        # Keep track of source documents
        source_documents = []
        
        # Stream the response token by token
        for message, metadata in self.app.graph.stream(input_message, config=self.config, stream_mode="messages"):
            # Check if this is a tool message (retrieval result)
            if message.type == "tool":
                parsed_results = self._parse_retrieved_text(message.content)
                source_documents = parsed_results
                
            # Yield a dictionary with the content, metadata, type, and source documents
            yield {
                "content": message.content,
                "metadata": metadata, 
                "type": message.type,
                "source_documents": source_documents if message.type == "tool" else []
            }

    def reset_conversation(self):
        """Reset the conversation state."""
        all_messages = self.app.graph.get_state(self.config).values["messages"]
        self.app.graph.update_state(self.config, {"messages": [RemoveMessage(id=m.id) for m in all_messages if m.id != self.system_prompt.id]})
    
    @staticmethod
    def _parse_retrieved_text(text):
        """
        Parse the formatted string with Source and Content sections into a list of dictionaries.
        
        Args:
            text (str): The formatted string with Source and Content sections
            
        Returns:
            list: A list of dictionaries, each containing source metadata and content
        """
        # Split the text into individual document entries
        entries = re.split(r'(?=Source: )', text.strip())
        
        results = []
        for entry in entries:
            if not entry.strip():
                continue
                
            # Extract source and content parts
            source_match = re.search(r'Source: (\{.*?\})', entry)
            content_match = re.search(r'Content: (.*?)(?=Source:|$)', entry, re.DOTALL)
            
            if source_match and content_match:
                # Parse the source dictionary string
                source_str = source_match.group(1)
                try:
                    # Convert string representation of dict to actual dict
                    source_dict = eval(source_str)
                except:
                    # Fallback if eval fails
                    source_dict = {"raw_source": source_str}
                    
                # Get the content
                content = content_match.group(1).strip()
                
                # Create a dictionary for this entry
                doc_dict = {
                    "source": source_dict,
                    "content": content
                }
                
                results.append(doc_dict)
        
        return results


# ==========================
# Example use
# ==========================
# if __name__ == "__main__":
#     chatbot = ChatbotService()

#     while True:
#         user_input = input("Enter a query (q to quit): ")
#         if user_input.lower() == "q":
#             break
#         if user_input.lower() == "r":
#             chatbot.reset_conversation()
#             print("Conversation reset.")
#             continue

#         print("\n--- Streaming Response ---")
#         # Track all sources to display at the end
#         all_sources = []
        
#         for message in chatbot.stream_message(user_input):
#             token = message["content"]
#             metadata = message["metadata"]
#             msg_type = message["type"]
#             source_docs = message["source_documents"]
            
#             # Save sources for later display
#             if source_docs and not all_sources:
#                 all_sources = source_docs
                
#             # Print token by token for AI responses
#             if msg_type == "tool":
#                 print("\n[Tool Call]")
#                 # print(token)
#             else:
#                 print(token, end="", flush=True)
                
#         print("\n")
        
#         # Display sources after the complete response
#         if all_sources:
#             print("\n--- Sources Used ---")
#             for i, doc in enumerate(all_sources):
#                 print(f"Source {i+1}:")
#                 print(f"  Metadata: {doc['source']}")
#                 # print(f"  Preview: {doc['content'][:100]}..." if len(doc['content']) > 100 else f"  Content: {doc['content']}")
#                 print(f"  Content: {doc['content']}")
#                 print()
        
        # # Alternative: Using send_message instead of stream_message
        # print("\n--- Non-streaming Response ---")
        # response = chatbot.send_message(user_input)
        # print(f"Content: {response['content']}")
        
        # if response['source_documents']:
        #     print("\nSources:")
        #     for i, doc in enumerate(response['source_documents']):
        #         print(f"Source {i+1}: {doc['source']['raw_source'] if 'raw_source' in doc['source'] else doc['source']}")
        
        # print("\n" + "-"*50 + "\n")