import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import time
import httpx
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from core.vector_store import VectorStoreManager
from config import Config as cfg

# ==========================
# Vector Store Manager
# ==========================
vector_store_manager = VectorStoreManager()
vector_store_manager.load_or_generate_vector_store()

# ==========================
# Exceptions
# ==========================
class APICallException(Exception):
    pass

# ==========================
# ChatGraph: Defines Conversation Flow
# ==========================
class ChatGraph:
    def __init__(self):
        """Initialize the conversation graph."""
        self.llm_client = init_chat_model(cfg.LLM_MODEL_NAME, model_provider=cfg.LLM_MODEL_PROVIDER)
        self.graph_builder = StateGraph(MessagesState)
        self._build_graph()

    def _build_graph(self):
        """Construct the state graph."""
        self.graph_builder.add_node(self.query_or_respond)
        self.graph_builder.add_node(self.get_tools())
        self.graph_builder.add_node(self.generate_response)

        self.graph_builder.set_entry_point("query_or_respond")
        self.graph_builder.add_conditional_edges(
            "query_or_respond", tools_condition, {END: END, "tools": "tools"}
        )
        self.graph_builder.add_edge("tools", "generate_response")
        self.graph_builder.add_edge("generate_response", END)

        self.graph = self.graph_builder.compile()
    
    def _safe_invoke(self, llm, input, max_retries=3):
        """Invoke the LLM with retries in case of network errors."""
        retries = 0
        while retries < max_retries:
            try:
                return llm.invoke(input)
            except httpx.HTTPStatusError as e:
                print(f"HTTP error: {e}. Retrying in 1s...")
            except httpx.RequestError as e:
                print(f"Network error: {e}. Retrying in 1s...")
            except Exception as e:
                print(f"Unexpected error: {e}")
                break  # Stop retrying for unknown errors

            retries += 1
            time.sleep(1)
        # Raise exception if max retries reached
        raise APICallException("Failed to invoke the language model after multiple retries.")

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(self, state: MessagesState):
        """Generate a response or call the retrieval tool if needed."""
        llm_with_tools = self.llm_client.bind_tools([self.retrieve])
        response = self._safe_invoke(llm_with_tools, state["messages"])
        return {"messages": [response]} if response else {"messages": []}

    # Step 2: Execute the retrieval.
    @staticmethod
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve relevant documents from the vector store."""
        retrieved_docs = vector_store_manager.vector_store.similarity_search(query, k=3)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}") for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    
    def get_tools(self):
        return ToolNode([self.retrieve])

    # Step 3: Generate a response using the retrieved content.
    def generate_response(self, state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)

        # TODO: Maybe loading a prompt template from langchain hub
        system_message_content = (
            "You are an assistant for question-answering tasks about the Natural Language Processing and Large Language Models course of"
            "University of Salerno. Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know without answering. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = self._safe_invoke(self.llm_client, prompt)
            
        return {"messages": [response]}