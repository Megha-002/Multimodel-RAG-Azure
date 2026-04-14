import os
from PIL import Image
import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = "tesseract"

# Load embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# ===============================
# PROCESS PDF
# ===============================

def process_pdf(file_path):
    print(f"📄 Processing PDF: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = text_splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks created")
    return chunks

# ===============================
# PROCESS IMAGE
# ===============================

def process_image(file_path):
    print(f"🖼️ Processing Image: {file_path}")
    img = Image.open(file_path)

    # Try OCR first
    ocr_text = pytesseract.image_to_string(img).strip()

    if ocr_text:
        print(f"   → Text found via OCR")
        chunks = text_splitter.split_documents([
            Document(page_content=ocr_text, metadata={"source": file_path})
        ])
        return chunks
    else:
        print(f"   → No text found in image, skipping")
        return []
# ===============================
# MAIN INGESTION PIPELINE
# ===============================

def ingest_folder(folder_path="data/"):
    all_chunks = []

    print(f"\n🚀 Starting ingestion from folder: {folder_path}\n")

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):
            chunks = process_pdf(file_path)
            all_chunks.extend(chunks)

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            chunks = process_image(file_path)
            all_chunks.extend(chunks)

        else:
            print(f"⚠️ Skipping unsupported file: {filename}")

    print(f"\n✅ Total chunks to store: {len(all_chunks)}")

    # Store all chunks in ChromaDB
    print("\n💾 Storing in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print("\n✅ Ingestion complete! ChromaDB is ready.")
    print(f"   Total documents stored: {len(all_chunks)}")

if __name__ == "__main__":
    ingest_folder("data/")