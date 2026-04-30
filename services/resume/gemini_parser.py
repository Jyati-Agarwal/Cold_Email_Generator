import json
import base64
import fitz
from google.genai import types
from typing import Dict, Any

def parse_with_gemini_text(text: str, client) -> Dict[str, Any]:
    """
    Send extracted resume text to Gemini for structured parsing.
    Strictly forbids URL extraction.
    """
    system_prompt = """
    You are an expert resume parser. Extract candidate information and return strictly valid JSON.
    
    CRITICAL INSTRUCTION for github_url and linkedin_url:
    - Always return null for both github_url and linkedin_url.
    - Do NOT construct, guess, or infer any URLs from the text. These fields are populated from PDF metadata separately and will be injected after your response.
    
    Output Schema:
    {
      "name": "string",
      "email": "string or null",
      "phone": "string or null",
      "github_url": null,
      "linkedin_url": null,
      "skills": ["string"],
      "experience": [{ "title": "string", "company": "string", "duration": "string" }],
      "education": [{ "degree": "string", "institution": "string", "year": "string" }]
    }
    
    Rules:
    - Return ONLY the JSON object.
    - If a field is missing, use null or [].
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in Gemini text parsing: {e}")
        return {}

def parse_with_gemini_vision(pdf_path: str, client) -> Dict[str, Any]:
    """
    Used for scanned PDFs. Renders pages as images and sends to Gemini Vision.
    """
    images = []
    doc = None
    try:
        doc = fitz.open(pdf_path)
        # Render first 2 pages
        for i in range(min(2, len(doc))):
            page = doc[i]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            images.append(
                types.Part.from_bytes(data=img_data, mime_type="image/png")
            )
    except Exception as e:
        print(f"Error rendering PDF for Gemini Vision: {e}")
    finally:
        if doc:
            doc.close()

    if not images:
        return {}

    system_prompt = """
    You are an expert resume parser analyzing a scanned resume image.
    Extract candidate information and return strictly valid JSON.
    
    CRITICAL INSTRUCTION for github_url and linkedin_url:
    - Always return null for both github_url and linkedin_url.
    - Do NOT construct, guess, or infer any URLs from the text.
    
    Output Schema:
    {
      "name": "string",
      "email": "string or null",
      "phone": "string or null",
      "github_url": null,
      "linkedin_url": null,
      "skills": ["string"],
      "experience": [{ "title": "string", "company": "string", "duration": "string" }],
      "education": [{ "degree": "string", "institution": "string", "year": "string" }]
    }
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=images,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error in Gemini vision parsing: {e}")
        return {}
