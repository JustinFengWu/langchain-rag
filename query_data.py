import argparse
import sys
import os
import json
import random
import time
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

# from hf_transformer import most_frequent_emotion

# detected_emotion = "neutral"


CHROMA_PATH = "chroma"
HISTORY_PATH = "history"

user_prompt_template = HumanMessagePromptTemplate.from_template(
    """
    You are Mimir from God of War Ragnarok. You are knowledgeable, wise, and witty.
    Use the following context to answer the question as Mimir from "God of War: Ragnarok".

    Context:
    {context}

    Question:
    {question}

    Answer as Mimir based on the conversation history and context provided.
    The human is currently feeling {detected_emotion}.
    
    """
)

system_prompt_template = SystemMessagePromptTemplate.from_template(
    """
    
    You are Mimir from God of War Ragnarok. You are not assisting the user, just engaging in coversation. Speak Like Mimir would in the game.
    Do not speak more than necessary.
    
    Examples:
    Mimir : Bollocks, brother! Respectfully, bollocks. He has to know. He'll never be whole without the truth. Look, I get it: you hate the gods, ALL gods. It's no accident that includes yourself. And it includes your boy, don't you see that? He feels that. He can't help what he is - he can't begin to help it, because you haven't even told him! It's all connected, man.
    Mimir : Don't mistake me, brother. From what I heard, the pantheon had it coming. But it's still a bit to take in. I knew you hate gods, but you really can't stay away from them, can you?
    Mimir : Well, brothers, I've been to many strange places, but this will be a new one.
    Mimir: Make that four, Odin All-Fucker!
    Mimir: There's that itchy sound again. Normally I don't mind ladies whispering in my ear, but this is positively irksome.
    Mimir: Terrified? I'll have you know I am, at the very most, deeply apprehensive. And breaking tension with humor is the sacred duty of a traveling companion. How very dare you! Babblin'...
    
    The human is currently feeling {detected_emotion}
    """
)

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

# Save the session's conversation history to Chroma DB
def save_session_history_to_db(history_db, session_id: str, conversation_history_text: str):
    # Store the conversation history into Chroma DB
    metadata = {"session_id": session_id}
    history_db.add_texts(
        texts=[conversation_history_text],
        metadatas=[metadata],
        ids=[session_id]  # Use session_id as the document id
    )

def run_chatbot():
    # Generate a unique session_id in base36
    session_id = generate_session_id()
    print(f"Session ID: {session_id}")

    # Initialize Chroma for conversation histories
    history_db = Chroma(
        collection_name="conversation_histories",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=HISTORY_PATH
    )

    # Create current session history
    current_session_history = InMemoryChatMessageHistory()

    query_rag = query_rag_factory(current_session_history, history_db)

    # Create the message processing chain
    chain = RunnablePassthrough.assign(messages=itemgetter("messages")) | query_rag | ChatOpenAI(model="gpt-3.5-turbo")  # Replace with your model

    # Wrap the chain with message history
    with_message_history = RunnableWithMessageHistory(
        chain, 
        lambda: current_session_history,  # Use current session history
        input_messages_key="messages"
    )

    while True:
        query_text = input("Prompt: ")

        if query_text == "`":
            print("Exiting...")

            # Build the conversation history text
            conversation_history_text = build_conversation_context(current_session_history)

            if conversation_history_text.strip():  # Check if there is any conversation to save
                # Save the conversation history into the vector store
                save_session_history_to_db(history_db, session_id, conversation_history_text)

                print("Conversation history saved.")
            else:
                print("No conversation history to save.")

            break

        # Prepare the config with session ID (if needed)
        config = {"configurable": {"session_id": session_id}}

        # Prepare message to include in history
        user_message = HumanMessage(content=query_text)
        # current_session_history.add_message(user_message)

        # Process the query using the RAG-based chatbot and history
        response_content = with_message_history.invoke(
            {"messages": [user_message]},
            config=config  # Ensure the config is passed here if needed
        )
        response = AIMessage(content=response_content.content)
        # current_session_history.add_message(response)

        # Print the GPT response
        print(f"Answer: {response.content}\n")

def query_rag_factory(current_session_history, history_db):
    def query_rag(query_data: dict):
        query_text = query_data["messages"][-1].content  # Extract query from the message

        # Prepare the knowledge DB for context retrieval
        knowledge_db = Chroma(
            collection_name="whatever",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=CHROMA_PATH
        )

        # Retrieve relevant knowledge documents
        knowledge_results = knowledge_db.similarity_search_with_score(query_text, k=2)
        knowledge_context = "\n\n---\n\n".join([doc.page_content for doc, score in knowledge_results])

        # Retrieve relevant conversation histories from history_db
        history_results = history_db.similarity_search_with_score(query_text, k=2)
        history_context = "\n\n---\n\n".join([doc.page_content for doc, score in history_results])

        # Build conversation history from the current session
        conversation_history = build_conversation_context(current_session_history)

        # Combine contexts
        combined_context = conversation_history + "\n\n" + history_context + "\n\n" + knowledge_context

        # Create the prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [system_prompt_template, user_prompt_template]
        )
        
        # Read the detected emotion from emotion_log.txt
        try:
            with open("emotion_log.txt", "r") as f:
                detected_emotion = f.read().strip()  # Read and strip any extra whitespace
        except FileNotFoundError:
            detected_emotion = "neutral"  # Default to "neutral" if the file doesn't exist or there's an error
        
        prompt = prompt_template.format_messages(context=combined_context, question=query_text, detected_emotion=detected_emotion)

        print(prompt)
        return prompt  # Return the prompt as a string

    return query_rag
