import time
import httpx
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from rag_chatbot.core.vector_store import VectorStoreManager
from langchain.chat_models import init_chat_model
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


graph_builder = StateGraph(MessagesState)

# model = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.3",
# )

# llm = ChatHuggingFace(llm=model)
llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
vm = VectorStoreManager()
vm.load_or_generate_vector_store()

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vm.vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def safe_invoke(llm, input, max_retries=3):
    '''Invoke the language model with retries in case of network errors.'''
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
            break  # If the error is unknown, stop retrying

        retries += 1
        time.sleep(1)  # Wait 1s before retrying


from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode


# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    response = safe_invoke(llm_with_tools, state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
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
    response = safe_invoke(llm, prompt)
        
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# Example usage:
if __name__ == "__main__":
    system_prompt = SystemMessage(
        '''You are an assistant for question-answering tasks about the Natural Language Processing and Large Language Models course of
        University of Salerno. Ask only question about this topic, use also the context (if available) retrieved from the documents to answer the question.
        If you don't know the answer, say that you don't know without answering. Don't try to answer out topic questions.
        Always retrieve information before answering if the query is about NLP and Large Language Models and the course.'''
    )
    conversation_state = {"messages": [system_prompt]}

    while True:
        query = input("Enter a query (q to quit): ")
        if query.lower() == "q":
            break
        
        # Aggiungi il nuovo messaggio dell'utente allo stato
        conversation_state["messages"].append({"role": "user", "content": query})

        # Esegui il chatbot con lo stato aggiornato
        for step in graph.stream(conversation_state, stream_mode="values"):
            response = step["messages"][-1]  # Ottieni l'ultima risposta
            response.pretty_print()  # Stampa la risposta in modo leggibile

        # Mantieni tutti i messaggi nella conversazione
        conversation_state["messages"].append(response)