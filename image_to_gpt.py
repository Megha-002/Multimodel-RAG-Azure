# ===============================
# STEP 1: IMPORT LIBRARIES
# ===============================

from PIL import Image
import pytesseract
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

# ===============================
# STEP 2: LOAD ENV VARIABLES
# ===============================

load_dotenv()

OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# ===============================
# STEP 3: IMAGE → TEXT FUNCTION
# ===============================

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# ===============================
# STEP 4: SEND TEXT TO GPT-4
# ===============================

def ask_gpt4(input_text):
    client = AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version=OPENAI_VERSION
    )

    response = client.chat.completions.create(
        model=OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {
                "role": "user",
                "content": f"The following text was extracted from an image. Please understand and respond:\n\n{input_text}"
            }
        ]
    )

    return response.choices[0].message.content

# ===============================
# STEP 5: MAIN FLOW (IMAGE → GPT)
# ===============================

if __name__ == "__main__":
    image_path = "images/image.png"

    print("🖼️ Reading image and extracting text...")
    extracted_text = extract_text_from_image(image_path)

    print("\n📄 Extracted Text:\n")
    print(extracted_text)

    if extracted_text.strip():
        print("\n🤖 Sending extracted text to GPT-4...")
        answer = ask_gpt4(extracted_text)

        print("\n💡 GPT-4 Answer:\n")
        print(answer)
    else:
        print("❌ No text found in image")
