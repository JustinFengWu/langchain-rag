import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage

def get_embedding_function():
    return OpenAIEmbeddings(model="text-embedding-ada-002")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# first fix the similarity search thing
# then fix model prompt [system, xxx, user, xxx]
def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    print(query_text)
    # Prepare the DB.
    db = Chroma(
        collection_name="whatever",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH  # This should automatically handles persistence
    )

    results = db.similarity_search_with_score(query_text, k=2)
    for res, score in results:
        print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")

    print(f"Number of results: {len(results)}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI(model="gpt-4o-mini")

    response_text = model.invoke([HumanMessage(content=prompt)])
    # print("\n\n")
    # print(response_text.content)

    sources = [doc.metadata.get("id", None) for doc, score in results]
    formatted_response = f"Response: {response_text.content}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()
