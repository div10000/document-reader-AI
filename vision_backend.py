import base64
import io
from PIL import Image
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

def encode_image(image_file):
    """Compresses the image and converts it to a Base64 string safely under 4MB."""
    # Open the image using Pillow
    img = Image.open(image_file)
    
    # Convert to RGB if it's an RGBA (transparent PNG) to save space
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Resize if the image is massive (helps stay under 4MB Base64 limit & 33MP limit)
    img.thumbnail((2000, 2000)) 
    
    # Save it to a temporary bytes buffer with compression
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    
    # Convert to Base64
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return base64_string

def ask_image(prompt_text, image_file):
    """Sends the image and prompt to the Groq Vision model."""
    
    # 1. Safely encode the image
    base64_image = encode_image(image_file)
    
    # Double check the 4MB limit (4MB = 4,194,304 bytes)
    if len(base64_image) > 4100000:
        return "⚠️ Error: The image is still too large after compression. Please upload a smaller image."

    # 2. Initialize the specific Scout Vision Model
    llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")
    
    # 3. Build the LangChain HumanMessage payload for vision
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )
    
    # 4. Invoke the model
    response = llm.invoke([message])
    return response.content