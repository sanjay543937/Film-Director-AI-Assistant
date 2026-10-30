
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "film_director_knowledge"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_vector_store():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB not found. Run the ingestion pipeline first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def retrieve_chunks(question: str, k: int = 5):
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        question,
        k=k,
    )

    retrieved_chunks = []

    for document, distance in results:
        retrieved_chunks.append({
            "text": document.page_content,
            "source": document.metadata.get("source", "unknown"),
            "page": document.metadata.get("page", 0),
            "distance": float(distance),
        })

    return retrieved_chunks


if __name__ == "__main__":
    question = input("Ask a filmmaking question: ")

    results = retrieve_chunks(question, k=5)

    for rank, item in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Distance: {item['distance']:.4f}")
        print(item["text"][:700])