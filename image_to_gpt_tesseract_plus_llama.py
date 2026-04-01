from PIL import Image
import pytesseract
import ollama
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ===============================
# IMAGE → TEXT FUNCTION
# ===============================

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# ===============================
# SEND TEXT TO LLAMA 3
# ===============================

def ask_llama(input_text):
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": f"The following text was extracted from an image. Please understand and respond:\n\n{input_text}"
            }
        ]
    )
    return response['message']['content']

# ===============================
# MAIN FLOW
# ===============================

if __name__ == "__main__":
    image_path = "images/image.png"

    print("🖼️ Reading image and extracting text...")
    extracted_text = extract_text_from_image(image_path)

    print("\n📄 Extracted Text:\n")
    print(extracted_text)

    if extracted_text.strip():
        print("\n🤖 Sending extracted text to LLaMA 3...")
        answer = ask_llama(extracted_text)

        print("\n💡 LLaMA 3 Answer:\n")
        print(answer)
    else:
        print("❌ No text found in image")