import sys
import os

# Add root directory to sys.path for proper module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_documents import load_all_documents


def create_vector_store():
    # Load documents (one page = one chunk)
    raw_docs = load_all_documents("data")

    # Split text into smaller chunks (for better embedding retrieval)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_chunks = []
    metadatas = []

    for doc in raw_docs:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            metadatas.append({
                "source": doc["source"],
                "page": doc["page"]
            })

    # Create embeddings
    embeddings = OpenAIEmbeddings()  # this will use OPENAI_API_KEY from env

    # Store vectors in ChromaDB
    Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory="vectorstore"
    )

    print(f"Created vector store with {len(all_chunks)} chunks.")

if __name__ == "__main__":
    create_vector_store()
