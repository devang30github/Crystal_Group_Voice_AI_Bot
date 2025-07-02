import fitz 
import easyocr
import io
from PIL import Image
import numpy as np 

pdf_path = 'data/Crystal-Logistics-Company-Profile.pdf' # <--- Replace with the path to your PDF
output_txt_path = 'data/extracted_text.txt' # <--- Define the output text file path

# Initialize EasyOCR reader. You can specify the languages you want to recognize.
reader = easyocr.Reader(['en'])

all_extracted_text = ""

try:
    doc = fitz.open(pdf_path) 
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Render the page to an image (pixmap).
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        
        # Convert pixmap to a PIL Image object
        img_pil = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Convert PIL Image to a NumPy array for EasyOCR
        img_np = np.array(img_pil) # <--- IMPORTANT CHANGE HERE
        
        # Perform OCR on the NumPy array image
        result = reader.readtext(img_np) # <--- Pass the NumPy array
        
        # Extract text and append to the overall text
        page_text = ""
        for (bbox, text, prob) in result:
            page_text += text + " "
        
        all_extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text.strip()}"
        
except FileNotFoundError:
    print(f"Error: PDF file not found at '{pdf_path}'")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'doc' in locals() and doc:
        doc.close() # Ensure the document is closed even if an error occurs

# Now, save the extracted text to a .txt file
try:
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(all_extracted_text)
    print(f"\nSuccessfully extracted text and saved to '{output_txt_path}'")
except Exception as e:
    print(f"Error saving text to file: {e}")

