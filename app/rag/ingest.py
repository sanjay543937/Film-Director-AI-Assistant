
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths are relative to the project root
PDF_DIR = Path("data/pdfs")
CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "film_director_knowledge"

# Small local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def extract_pdf_pages():
    """Extract text from every PDF, preserving page metadata."""
    documents = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {PDF_DIR.resolve()}. "
            "Put your filmmaking PDF inside data/pdfs/."
        )

    for pdf_path in pdf_files:
        print(f"Reading PDF: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            # Normalize whitespace without removing punctuation.
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise ValueError(
            "No extractable text found. The PDF may be scanned "
            "and require OCR."
        )

    return documents


def build_vector_database():
    """Split extracted pages, embed chunks and store them in Chroma."""
    pages = extract_pdf_pages()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text chunks were created.")

    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Replace the previous collection so rerunning ingestion
    # does not append duplicate chunks.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Deleted previous collection.")
    except Exception:
        # It is normal for the collection not to exist initially.
        pass

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"Stored chunks in ChromaDB: {len(chunks)}")
    print(f"Database directory: {CHROMA_DIR.resolve()}")

    return vector_store


if __name__ == "__main__":
    build_vector_database()