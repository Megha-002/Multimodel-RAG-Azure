import os
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


# ── Type 1: Regular PDFs (text-based) ─────────────────────
def load_pdf_file(file_path: str) -> dict:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text = ""
    for page in pages:
        if page.page_content:
            text += page.page_content + "\n"

    return {"text": text.strip(), "source": Path(file_path).name}


# ── Type 2: Scanned PDFs or Images with text (OCR) ────────
def load_image_with_ocr(file_path: str) -> dict:
    import pytesseract
    from PIL import Image

    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)

    return {"text": text.strip(), "source": Path(file_path).name}


def load_scanned_pdf(file_path: str) -> dict:
    """Converts each PDF page to image using pymupdf then runs OCR on it."""
    import fitz  # pymupdf
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        # Convert page to image at 300 DPI for good OCR quality
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(img)
        if page_text:
            text += page_text + "\n"

    doc.close()
    return {"text": text.strip(), "source": Path(file_path).name}


# ── Type 3: Images with no text (Vision model) ────────────
def describe_image_with_vision(file_path: str) -> dict:
    """Sends image to Groq Vision API to get a text description."""
    from PIL import Image
    import io

    # Convert any format (webp, bmp, etc.) to PNG
    img = Image.open(file_path)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in detail. Include all visible text, objects, charts, diagrams, and any meaningful information."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }

    r = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=60)

    # Print actual error before crashing
    if r.status_code != 200:
        print(f"     🔴 Groq Vision Error: {r.status_code}")
        print(f"     🔴 Response: {r.text}")
        r.raise_for_status()

    description = r.json()["choices"][0]["message"]["content"]
    return {"text": description.strip(), "source": Path(file_path).name}

# ── Smart Loader: Decides which method to use ─────────────
def has_extractable_text(file_path: str) -> bool:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # If very little text found, likely scanned
    return len(text.strip()) > 50


def image_has_text(file_path: str) -> bool:
    import pytesseract
    from PIL import Image

    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return len(text.strip()) > 20


def load_all_documents(data_folder: str) -> list:

    documents = []
    supported_images = (".png", ".jpg", ".jpeg", ".tiff", ".bmp",".webp")

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        ext = Path(file_name).suffix.lower()

        try:
            # ── PDF files ──────────────────────────────────
            if ext == ".pdf":
                if has_extractable_text(file_path):
                    print(f"  📄 PDF (text): {file_name}")
                    doc = load_pdf_file(file_path)
                else:
                    print(f"  🔍 PDF (scanned → OCR): {file_name}")
                    doc = load_scanned_pdf(file_path)

                if doc["text"]:
                    documents.append(doc)
                    print(f"     ✅ Loaded successfully")
                else:
                    print(f"     ⚠️ No text extracted")

            # ── Image files ────────────────────────────────
            elif ext in supported_images:
                if image_has_text(file_path):
                    print(f"  🖼️ Image (has text → OCR): {file_name}")
                    doc = load_image_with_ocr(file_path)
                else:
                    print(f"  🎨 Image (no text → Vision AI): {file_name}")
                    doc = describe_image_with_vision(file_path)

                if doc["text"]:
                    documents.append(doc)
                    print(f"     ✅ Loaded successfully")
                else:
                    print(f"     ⚠️ No content extracted")

            # ── Text files ─────────────────────────────────
            elif ext == ".txt":
                print(f"  📝 Text file: {file_name}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append({"text": content, "source": file_name})
                print(f"     ✅ Loaded successfully")

            else:
                print(f"  ⏭️ Skipped: {file_name} (unsupported)")

        except Exception as e:
            print(f"  ❌ Error processing {file_name}: {str(e)}")

    return documents