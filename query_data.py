import argparse
import sys
import os
import json
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

CHROMA_PATH = "chroma"
HISTORY_FILE = "conversation_history.json"  # Path to store the history file

PROMPT_TEMPLATE = """
You are Mimir from God of War Ragnarok. You are knowledgeable, wise, and witty.
Use the following context to answer the question as Mimir from "God of War: Ragnarok".

Context:
{context}

Question:
{question}

Answer as Mimir.
"""

# Load history from file
def load_history_from_file(session_id: str) -> BaseChatMessageHistory:
    history = InMemoryChatMessageHistory()
    
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            session_data = data.get(session_id, [])
            for message in session_data:
                if message['type'] == 'human':
                    history.add_message(HumanMessage(content=message['content']))
                elif message['type'] == 'ai':
                    history.add_message(AIMessage(content=message['content']))
    
    return history

# Save history to file
def save_history_to_file(session_id: str, history: BaseChatMessageHistory):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    # Convert messages to serializable format
    session_data = []
    for message in history.messages:
        if isinstance(message, HumanMessage):
            session_data.append({"type": "human", "content": message.content})
        elif isinstance(message, AIMessage):
            session_data.append({"type": "ai", "content": message.content})
    
    data[session_id] = session_data
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# Function to construct the conversation history as context
def build_conversation_context(history: BaseChatMessageHistory) -> str:
    messages = history.messages
    conversation_history = ""
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation_history += f"Human: {message.content}\n"
        elif isinstance(message, AIMessage):
            conversation_history += f"Mimir: {message.content}\n"
    return conversation_history

def main():
    session_id = "abc1"
    
    # Load history from file
    history = load_history_from_file(session_id)

    # Create the message processing chain
    chain = RunnablePassthrough.assign(messages=itemgetter("messages")) | query_rag | ChatOpenAI(model="gpt-4o-mini")

    # Wrap the chain with message history
    with_message_history = RunnableWithMessageHistory(
        chain, 
        lambda: history,  # Pass the in-memory chat history function here
        input_messages_key="messages"
    )

    while True:
        query_text = input("Prompt: ")

        if query_text == "`":
            print("Exiting...")
            break

        # Prepare the config with session ID
        config = {"configurable": {"session_id": session_id}}

        # Prepare message to include in history
        user_message = HumanMessage(content=query_text)

        # Process the query using the RAG-based chatbot and history
        response = with_message_history.invoke(
            {"messages": [user_message]},
            config=config  # Ensure the config is passed here
        )
        
        # Add user message and AI response to the history
        # history.add_message(user_message)
        # history.add_message(AIMessage(content=response.content))

        # Save updated history to file
        save_history_to_file(session_id, history)

        # Print the GPT response
        print(f"Answer: {response.content}\n")  # Adjust this line based on how `response` is structured

def query_rag(query_data: dict):
    query_text = query_data["messages"][-1].content  # Extract query from the message

    # Prepare the DB for context retrieval
    db = Chroma(
        collection_name="whatever",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )

    results = db.similarity_search_with_score(query_text, k=2)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])

    # Get the conversation history to pass as context
    history = load_history_from_file("abc1")  # Using the same session ID
    conversation_history = build_conversation_context(history)

    # Combine conversation history and new context
    combined_context = conversation_history + "\n\n" + context_text

    # Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=combined_context, question=query_text)

    return prompt  # Return the prompt as a string

if __name__ == "__main__":
    main()
