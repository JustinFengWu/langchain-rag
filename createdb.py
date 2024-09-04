import os
import openai
import shutil
import uuid 

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=150,
        length_function=len,
        add_start_index=True,
    )

    for i, doc in enumerate(documents):
        if "id" not in doc.metadata:
            # Generate a unique id using uuid and assign it to the document
            doc.metadata["id"] = str(uuid.uuid4()) 
            
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    for i, chunk in enumerate(chunks):
        # Assign chunk-specific ids based on the document id and chunk number
        chunk.metadata["chunk_id"] = f"{chunk.metadata['id']}_chunk_{i}"

    document = chunks[2]
    print(document.page_content)
    print("\n\n\n")
    print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma(
        collection_name="whatever",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH  # This should automatically handles persistence
    )

    db.add_documents(chunks)


if __name__ == "__main__":
    main()
    