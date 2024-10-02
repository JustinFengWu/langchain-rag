import argparse
import sys
import os
import json
import random
import time
from uuid import uuid4
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

# Base36 encoding function
def base36encode(number, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    if not isinstance(number, int):
        raise TypeError('number must be an integer')
    base36 = ''
    sign = ''
    if number < 0:
        sign = '-'
        number = -number
    if 0 <= number < len(alphabet):
        return sign + alphabet[number]
    while number != 0:
        number, i = divmod(number, len(alphabet))
        base36 = alphabet[i] + base36
    return sign + base36

# Generate a unique session_id
def generate_session_id():
    # Use current time in nanoseconds and a random number for uniqueness
    random_number = time.time_ns() + random.getrandbits(64)
    session_id = base36encode(random_number)
    return session_id

# Load all summaries from file
def load_all_summaries() -> BaseChatMessageHistory:
    history = InMemoryChatMessageHistory()
    
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
            for session_id, session_data in data.items():
                if "summary" in session_data:
                    # If summary exists, add it as an AIMessage
                    history.add_message(AIMessage(content=session_data["summary"]))
    return history

# Save history to file
def save_history_to_file(session_id: str, history: BaseChatMessageHistory = None, summary: str = None):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    
    if summary is not None:
        data[session_id] = {"summary": summary}
    elif history is not None:
        # Convert messages to serializable format
        session_data = []
        for message in history.messages:
            if isinstance(message, HumanMessage):
                session_data.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                session_data.append({"type": "ai", "content": message.content})
        
        data[session_id] = {"messages": session_data}
        
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

# Combined history class
class CombinedChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, histories):
        self.histories = histories  # List of BaseChatMessageHistory

    @property
    def messages(self):
        combined_messages = []
        for h in self.histories:
            combined_messages.extend(h.messages)
        return combined_messages

    def add_message(self, message):
        # Add message to the last history (current session)
        self.histories[-1].add_message(message)

    def clear(self):
        # Clear messages from the current session history
        self.histories[-1].clear()

def main():
    # Generate a unique session_id in base36
    session_id = generate_session_id()
    print(f"Session ID: {session_id}")
    
    # Load all summaries from file
    history = load_all_summaries()  # Contains previous summaries

    # Create current session history
    current_session_history = InMemoryChatMessageHistory()

    # Combine histories
    combined_history = CombinedChatMessageHistory([history, current_session_history])
    
    query_rag = query_rag_factory(combined_history)

    # Create the message processing chain
    chain = RunnablePassthrough.assign(messages=itemgetter("messages")) | query_rag | ChatOpenAI(model="gpt-4o-mini")

    # Wrap the chain with combined message history
    with_message_history = RunnableWithMessageHistory(
        chain, 
        lambda: combined_history,  # Use combined history here
        input_messages_key="messages"
    )

    while True:
        query_text = input("Prompt: ")

        if query_text == "`":
            print("Exiting...")

            # Summarize the current session's conversation history
            conversation_history_text = build_conversation_context(current_session_history)

            if conversation_history_text.strip():  # Check if there is any conversation to summarize
                # Create summarization prompt
                summarization_prompt = f"Summarize all information below by keeping all critical information:\n\n{conversation_history_text}"

                # Send summarization prompt to the model
                summarization_response = ChatOpenAI(model="gpt-4o-mini").invoke([HumanMessage(content=summarization_prompt)])

                summary_text = summarization_response.content

                # Save the summary back into the .json file, under the current session_id
                save_history_to_file(session_id, summary=summary_text)

                print("Conversation summary saved.")
            else:
                print("No conversation history to summarize.")

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
        
        # Save updated current session history to file
        save_history_to_file(session_id, history=current_session_history)

        # Print the GPT response
        print(f"Answer: {response.content}\n")  # Adjust this line based on how `response` is structured

def query_rag_factory(combined_history):
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

        # Use the combined history
        conversation_history = build_conversation_context(combined_history)

        # Combine conversation history and new context
        combined_context = conversation_history + "\n\n" + context_text

        # Create the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=combined_context, question=query_text)

        return prompt  # Return the prompt as a string

    return query_rag

if __name__ == "__main__":
    main()
