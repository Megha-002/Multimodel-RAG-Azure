from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def chunk_documents(documents: list, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """
    Takes a list of {text, source} documents and returns
    a list of {text, source} chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Splitting priority: paragraphs → sentences → words → characters
    """
    chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 50))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": doc["source"]
            })

    print(f"  📦 Total chunks created: {len(all_chunks)}")
    return all_chunks