import requests
from PIL import Image
from io import BytesIO

# ===============================
# TEXT → IMAGE FUNCTION
# ===============================

def generate_image(prompt_text):
    print("\n🎨 Generating image...")
    
    # Free image generation API - no key needed
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt_text)}"
    
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # Save image
    image = Image.open(BytesIO(response.content))
    image_path = "generated_image.png"
    image.save(image_path)
    
    return image_path

# ===============================
# MAIN FLOW
# ===============================

if __name__ == "__main__":
    prompt = input("📝 Enter text to generate image: ")
    
    image_path = generate_image(prompt)
    
    print(f"\n✅ Image generated and saved as: {image_path}")
    
    # Show the image
    img = Image.open(image_path)
    img.show()