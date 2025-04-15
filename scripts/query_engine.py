from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def load_vector_store():
    return Chroma(
        persist_directory="vectorstore",
        embedding_function=OpenAIEmbeddings()
    )

def ask_question(question, k=1):
    db = load_vector_store()
    results = db.similarity_search(question, k=k)

    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(res.page_content)
        print(f"Source: {res.metadata.get('source')} | Page: {res.metadata.get('page')}")


if __name__ == "__main__":
    ask_question("How do I start the engine?")
