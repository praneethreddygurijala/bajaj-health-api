from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Any
import google.generativeai as genai
import requests
from PIL import Image
import io
import os
import json
import re
import cv2
import numpy as np
from dotenv import load_dotenv

# Try importing pytesseract, but don't fail if missing
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except:
    TESSERACT_AVAILABLE = False
    print("⚠️ Tesseract not available - using Gemini Vision only")

# 1. Load Environment Variables
load_dotenv()

app = FastAPI(title="Bajaj Health Datathon API", version="4.0.0")

# 2. Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set in .env file!")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Use the latest fast model
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# --- Data Models (Strictly matching New Problem Statement) ---

class DocumentRequest(BaseModel):
    document: HttpUrl

class TokenUsage(BaseModel):
    total_tokens: int
    input_tokens: int
    output_tokens: int

class BillItem(BaseModel):
    item_name: str
    item_amount: float
    item_rate: float
    item_quantity: float

class PageWiseLineItem(BaseModel):
    page_no: str
    page_type: str  # New Requirement: "Bill Detail | Final Bill | Pharmacy"
    bill_items: List[BillItem]

class ExtractionData(BaseModel):
    pagewise_line_items: List[PageWiseLineItem]
    total_item_count: int

class BillExtractionResponse(BaseModel):
    is_success: bool
    token_usage: TokenUsage
    data: ExtractionData

# --- Core Logic Functions ---

def download_document(url: str) -> bytes:
    """Download document with timeout safety"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")

def preprocess_image(image_bytes: bytes) -> tuple:
    """Pre-processing for Images to improve OCR/Vision"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too massive
        width, height = image.size
        if max(width, height) > 3000:
            scale = 3000 / max(width, height)
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Contrast Enhancement (CLAHE)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        enhanced = Image.fromarray(thresh)
        
        return enhanced, image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

def perform_ocr(image: Image.Image) -> str:
    """Tesseract OCR Wrapper (Optional Helper)"""
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(image)
    except:
        return "" 

def analyze_with_gemini(content_part: Any, mime_type: str, ocr_context: str = "") -> tuple:
    """
    The Brain: Handles Extraction, Classification, and Token Counting
    """
    prompt = f"""
    You are an expert medical billing auditor. Extract structured data from this document.
    
    CONTEXT:
    {ocr_context if ocr_context else "No OCR context available - analyze image directly"}

    ### CRITICAL RULES:
    1. **Page Classification:** For each page, determine `page_type`. It MUST be exactly one of: "Bill Detail", "Final Bill", or "Pharmacy".
    2. **Extraction:** Extract Name, Rate, Qty, Amount for every line item.
    3. **No Totals:** Do NOT extract "Sub Total", "Total", or "Grand Total" as items.
    4. **Multi-Page:** Group items by page number accurately.
    5. **Math Check:** Ensure item_amount is roughly rate * quantity.

    ### OUTPUT JSON STRUCTURE:
    {{
      "pagewise_line_items": [
        {{
          "page_no": "1",
          "page_type": "Bill Detail", 
          "bill_items": [
            {{
              "item_name": "Service Name",
              "item_amount": 0.00,
              "item_rate": 0.00,
              "item_quantity": 0.00
            }}
          ]
        }}
      ]
    }}
    """

    try:
        # Prepare content for Gemini
        generation_config = {"response_mime_type": "application/json"}
        
        if mime_type == "application/pdf":
            response = model.generate_content([prompt, {"mime_type": mime_type, "data": content_part}], generation_config=generation_config)
        else:
            response = model.generate_content([prompt, content_part], generation_config=generation_config)

        # 1. Parse JSON Response
        response_text = response.text.strip()
        data = json.loads(response_text)
        
        # 2. Validation & Noise Filter
        total_count = 0
        
        if "pagewise_line_items" in data:
            for page in data["pagewise_line_items"]:
                valid_items = []
                # Ensure page_type is valid
                if page.get("page_type") not in ["Bill Detail", "Final Bill", "Pharmacy"]:
                    page["page_type"] = "Bill Detail" # Default fallback
                
                if "bill_items" in page:
                    for item in page["bill_items"]:
                        # Clean and cast numbers
                        try:
                            amt = float(item.get("item_amount", 0))
                            item["item_amount"] = amt
                            item["item_rate"] = float(item.get("item_rate", 0))
                            item["item_quantity"] = float(item.get("item_quantity", 0))
                            
                            # Noise Filter: Ignore 0.00 items
                            if amt > 0:
                                total_count += 1
                                valid_items.append(item)
                        except:
                            continue # Skip bad items
                            
                page["bill_items"] = valid_items

        data["total_item_count"] = total_count
        
        # 3. Extract Token Usage
        # Gemini 2.0 Flash returns usage_metadata
        token_usage = {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0
        }
        
        if hasattr(response, "usage_metadata"):
            token_usage = {
                "total_tokens": response.usage_metadata.total_token_count,
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count
            }
        
        return data, token_usage

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

# --- API Endpoints ---

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Bajaj Health Datathon API",
        "version": "4.0.0",
        "tesseract_available": TESSERACT_AVAILABLE
    }

@app.post("/extract-bill-data", response_model=BillExtractionResponse)
async def extract_bill_data(request: DocumentRequest):
    try:
        # 1. Download
        doc_bytes = download_document(str(request.document))
        
        # 2. Detect File Type
        is_pdf = doc_bytes.startswith(b'%PDF')
        
        if is_pdf:
            # --- PDF FLOW ---
            data, token_usage = analyze_with_gemini(doc_bytes, "application/pdf")
        else:
            # --- IMAGE FLOW ---
            enhanced_img, original_img = preprocess_image(doc_bytes)
            ocr_text = perform_ocr(enhanced_img)
            data, token_usage = analyze_with_gemini(original_img, "image/jpeg", ocr_text)

        # 3. Return Standard Response
        return BillExtractionResponse(
            is_success=True,
            token_usage=TokenUsage(**token_usage),
            data=ExtractionData(**data)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)