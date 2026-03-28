# STEP 1: Import required libraries
from PIL import Image
import pytesseract

# STEP 2: Path to image
image_path = "images/image.png"

# STEP 3: Open the image
img = Image.open(image_path)

# STEP 4: Extract text using OCR
extracted_text = pytesseract.image_to_string(img)

# STEP 5: Print extracted text
print("\n🖼️ Extracted Text from Image:\n")
print(extracted_text)
