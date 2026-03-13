import os
import json
import base64
import tempfile
import fitz  # PyMuPDF - used for converting PDF to images
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from dotenv import load_dotenv

# Load environment variables (for local testing)
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="Israeli Invoice OCR API",
    description="API that extracts structured JSON from Israeli invoices using Azure Document Intelligence and GPT-4o Vision.",
    version="1.0.0",
)

# Initialize clients securely using environment variables
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# בדיקה מוקדמת כדי לזהות שגיאות מיד בעליית השרת ולא רק בשליחת בקשה
if not all([AZURE_ENDPOINT, AZURE_KEY, OPENAI_API_KEY]):
    print("❌ CRITICAL ERROR: Missing environment variables! Check Railway settings.")

# הגדרה ישירה של ה-clients - ללא try/except, כך שיהיו מוגדרים גלובלית
azure_client = DocumentAnalysisClient(
    endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY)
)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def convert_pdf_to_base64_images(file_path):
    """
    Helper function that converts each PDF page to a Base64 encoded JPEG image.
    """
    base64_images = []
    pdf_document = fitz.open(file_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        # Zoom by a factor of 2 for high resolution
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("jpeg")
        base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
        base64_images.append(base64_encoded)

    pdf_document.close()
    return base64_images


def analyze_israeli_invoice(file_path):
    # ==========================================
    # 1. Extract raw text using Azure
    # ==========================================
    with open(file_path, "rb") as f:
        poller = azure_client.begin_analyze_document("prebuilt-invoice", f)
        result = poller.result()

    raw_text = result.content
    if not raw_text:
        return None

    # ==========================================
    # 2. Prepare images for OpenAI's Vision model
    # ==========================================
    base64_images = convert_pdf_to_base64_images(file_path)

    # ==========================================
    # 3. Hybrid processing (Vision + Text) using OpenAI
    # ==========================================
    system_prompt = """
    You are an expert Israeli accountant and a highly precise data extractor. 
    You are operating in a HYBRID MULTIMODAL mode. I am providing you with BOTH:
    1. The visual images of the invoice.
    2. The raw OCR text extracted from this invoice by an external engine.
    
    CRITICAL HYBRID INSTRUCTIONS:
    - USE THE IMAGE as your single source of truth for the physical layout, row order, and determining WHICH price belongs to WHICH description. The OCR text often scrambles the Z-order.
    - USE THE RAW OCR TEXT as your single source of truth for exact spelling, long ID numbers (ח.פ/ע.מ), invoice numbers, and accurate digits to prevent visual hallucinations.
    
    CRITICAL LOGICAL RULES FOR ISRAELI INVOICES:
    1. Customer Name: MUST contain letters. Look near words like "לכבוד". Ignore random numbers near it.
    
    1.a. Customer HP / Tax ID: The 'customer_hp' is the 9-digit Israeli company/dealer number of the BUYER. 
         - Look for it near the customer's name, OR anywhere else in the document labeled as "ח.פ", "ע.מ", "מס. חברה לקוח", or "תיק מע\"מ לקוח".
         - CRITICAL: Do NOT confuse the customer's HP with the Vendor's HP (the seller's ID, which usually appears at the very top of the page).
    
    2. STRICT ROW ALIGNMENT & VALUE PAIRING: Ensure price, description, and quantity actually belong to THAT exact physical row in the image. Do NOT shift prices! Do NOT mistake the invoice's overall Subtotal (סה"כ) for the price of an item!
       
    3. Global vs. Line Item Discount: Line Discount maps to `discount_amount` in the items array. Global Discount (bottom of invoice) maps ONLY to `total_discount` in the summary. NEVER inject global discounts into individual items.
       
    4. Mathematical Verification: 
       - Row Level: Quantity * (Unit Price Before Discount - Discount Amount) MUST equal the Line Total.
       - Summary Level: Subtotal Before Discount - Total Discount + Tax Amount MUST equal Total Including Tax.
       
    5. Row Order: Output the `items` array ordered logically from top to bottom based on the IMAGE.
    
    6. Allocation Number: If 'allocation_number' (הקצאה / קמאה) exists, it is a 9+ digit number.
    
    7. ZERO-VALUE ITEMS: Extract EVERY single line item present in the table. Do not skip items even if their price is 0.00.
    
    8. DYNAMIC EXTRA COLUMNS: Extract extra columns into the `extra_columns` object (e.g. "ת.מ" -> "delivery_note", "רכב" -> "vehicle_number").
       
    9. DATE PARSING: Format as YYYY-MM-DD. "26" as a year means 2026.
    
    You MUST return ONLY a valid JSON object matching this exact structure:
    {
        "header": {
            "customer_name": string or null,
            "customer_hp": string or null,
            "invoice_number": string or null,
            "invoice_type": string,
            "invoice_date": string (YYYY-MM-DD),
            "allocation_number": string or null
        },
        "items": [
            {
                "row_number": int,
                "sku": string or null,
                "quantity": float,
                "description": string,
                "unit_price_before_discount": float,
                "discount_amount": float,
                "unit_price_after_discount": float,
                "line_total": float,
                "extra_columns": {}
            }
        ],
        "summary": {
            "subtotal_before_tax_and_discount": float,
            "total_discount": float,
            "subtotal_after_discount": float,
            "tax_amount": float,
            "total_including_tax": float,
            "tax_percentage": int
        }
    }
    """

    user_content = [
        {
            "type": "text",
            "text": f"Here is the raw OCR text extracted by Azure:\n\n{raw_text}\n\nAnd here are the visual images of the invoice. Use them to ensure correct row alignment!",
        }
    ]

    for base64_img in base64_images:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_img}",
                    "detail": "high",
                },
            }
        )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        gpt_output = response.choices[0].message.content
        final_json = json.loads(gpt_output)

        return final_json

    except Exception as e:
        print(f"❌ Error communicating with OpenAI API: {e}")
        return None


# ==========================================
# API ENDPOINTS
# ==========================================


@app.get("/")
def read_root():
    """
    Health check endpoint to ensure the API is running.
    """
    return {
        "status": "online",
        "message": "AI Invoice Extractor API is running.",
        "documentation_url": "/docs",
    }


@app.post("/extract-invoice/")
async def extract_invoice_endpoint(file: UploadFile = File(...)):
    """
    Endpoint that accepts a PDF file and returns the extracted JSON data.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_path = tmp_file.name

    try:
        # Process the file
        parsed_data = analyze_israeli_invoice(tmp_path)

        if parsed_data is None:
            raise HTTPException(
                status_code=500, detail="Failed to extract data from the document."
            )

        return JSONResponse(content=parsed_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# This allows Railway or local machines to run the app directly
if __name__ == "__main__":
    # Uses the PORT environment variable if available (required for Railway), defaults to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
