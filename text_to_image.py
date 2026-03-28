# # ===============================
# # STEP 1: IMPORT LIBRARIES
# # ===============================

# from openai import AzureOpenAI
# from dotenv import load_dotenv
# import os
# import base64

# # ===============================
# # STEP 2: LOAD ENV VARIABLES
# # ===============================

# load_dotenv()

# OPENAI_KEY = os.getenv("AZURE_DALLE_OPENAI_API_KEY")
# OPENAI_ENDPOINT = os.getenv("AZURE_DALLE_OPENAI_ENDPOINT")
# OPENAI_VERSION = os.getenv("AZURE_DALLE_OPENAI_API_VERSION")
# DALLE_DEPLOYMENT = os.getenv("AZURE_DALLE_DEPLOYMENT_NAME")

# # ===============================
# # STEP 3: CREATE AZURE OPENAI CLIENT
# # ===============================

# client = AzureOpenAI(
#     api_key=OPENAI_KEY,
#     azure_endpoint=OPENAI_ENDPOINT,
#     api_version=OPENAI_VERSION
# )

# # ===============================
# # STEP 4: TEXT → IMAGE FUNCTION
# # ===============================

# def generate_image(prompt_text):
#     result = client.images.generate(
#         model=DALLE_DEPLOYMENT,
#         prompt=prompt_text,
#         size="1024x1024"
#     )

#     image_base64 = result.data[0].b64_json
#     image_bytes = base64.b64decode(image_base64)

#     with open("generated_image.png", "wb") as f:
#         f.write(image_bytes)

#     return "generated_image.png"

# # ===============================
# # STEP 5: MAIN FLOW
# # ===============================

# if __name__ == "__main__":
#     user_prompt = input("📝 Enter text to generate image: ")

#     print("\n🎨 Generating image...")
#     image_path = generate_image(user_prompt)

#     print(f"\n✅ Image generated and saved as: {image_path}")

# =========================================
# TEXT-TO-IMAGE USING AZURE DALL·E
# =========================================

from openai import AzureOpenAI
from dotenv import load_dotenv
import os
import requests

# =========================================
# STEP 1: LOAD ENV VARIABLES
# =========================================

load_dotenv()

OPENAI_KEY = os.getenv("AZURE_DALLE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_DALLE_OPENAI_ENDPOINT")  # base URL
OPENAI_VERSION = os.getenv("AZURE_DALLE_OPENAI_API_VERSION")
DALLE_DEPLOYMENT = os.getenv("AZURE_DALLE_DEPLOYMENT_NAME")

# =========================================
# STEP 2: CREATE AZURE OPENAI CLIENT
# =========================================

client = AzureOpenAI(
    api_key=OPENAI_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version=OPENAI_VERSION
)

# =========================================
# STEP 3: FUNCTION TO GENERATE IMAGE
# =========================================

def generate_image(prompt_text):
    # Call Azure OpenAI DALL·E deployment
    result = client.images.generate(
        model=DALLE_DEPLOYMENT,
        prompt=prompt_text,
        size="1024x1024"  # You can also use "512x512"
    )

    # Azure returns an image URL
    image_url = result.data[0].url
    print(f"Image URL: {image_url}")  # optional for debugging

    # Download image
    response = requests.get(image_url)
    response.raise_for_status()  # stop if download fails

    # Save locally
    image_path = "generated_image.png"
    with open(image_path, "wb") as f:
        f.write(response.content)

    return image_path

# =========================================
# STEP 4: MAIN FLOW
# =========================================

if __name__ == "__main__":
    prompt = input("📝 Enter text to generate image: ")

    print("\n🎨 Generating image...")
    image_path = generate_image(prompt)

    print(f"\n✅ Image generated and saved as: {image_path}")
# meghna git hub