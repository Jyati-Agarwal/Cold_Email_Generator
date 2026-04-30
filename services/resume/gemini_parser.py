import json
import base64
import fitz
from google.genai import types
from typing import Dict, Any

def parse_with_gemini_text(text: str, client) -> Dict[str, Any]:
    """
    Send extracted resume text to Gemini for structured parsing.
    Deterministic code enriches links after this step, so the model focuses on
    structured resume evidence instead of guessing URLs.
    """
    system_prompt = """
You are an expert resume parser. Extract every useful factual detail from the
resume text and return strictly valid JSON.

Important:
- Do not summarize away achievements, metrics, technologies, project names, or
  leadership/scope details. These are needed to write a high-quality application
  email later.
- Do not invent or infer missing facts.
- For URL fields, return null/[] unless the exact URL is visibly present in the
  text. A deterministic PDF/text extractor will populate embedded links later.

Return ONLY a JSON object matching this schema:
{
  "name": "string or null",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "headline": "string or null",
  "summary": "string or null",
  "total_experience": "string or null",
  "github_url": "string or null",
  "linkedin_url": "string or null",
  "portfolio_url": "string or null",
  "other_links": ["string"],
  "skills": ["string"],
  "experience": [
    {
      "title": "string or null",
      "company": "string or null",
      "location": "string or null",
      "duration": "string or null",
      "achievements": ["string"],
      "technologies": ["string"]
    }
  ],
  "projects": [
    {
      "name": "string or null",
      "description": "string or null",
      "tech_stack": ["string"],
      "achievements": ["string"],
      "url": "string or null"
    }
  ],
  "education": [
    {
      "degree": "string or null",
      "institution": "string or null",
      "year": "string or null",
      "details": ["string"]
    }
  ],
  "certifications": ["string"],
  "awards": ["string"],
  "publications": ["string"],
  "languages": ["string"]
}

Rules:
- Preserve exact names, phone numbers, emails, company names, project names, and
  numeric metrics as written.
- Extract total years of experience only when explicitly stated in the resume.
- Split skills into clean individual skill strings.
- Keep achievement bullets factual and concise.
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
        # Render the first 3 pages at higher resolution for better OCR.
        zoom = fitz.Matrix(2, 2)
        for i in range(min(3, len(doc))):
            page = doc[i]
            pix = page.get_pixmap(matrix=zoom, alpha=False)
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
You are an expert resume parser analyzing scanned resume images. Extract every
useful factual detail and return strictly valid JSON.

Important:
- Preserve exact names, phone numbers, emails, visible URLs, project names,
  metrics, technologies, and achievement details.
- Do not invent or infer missing facts.
- Only return a URL if it is visibly present and legible.

Return ONLY a JSON object matching this schema:
{
  "name": "string or null",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "headline": "string or null",
  "summary": "string or null",
  "total_experience": "string or null",
  "github_url": "string or null",
  "linkedin_url": "string or null",
  "portfolio_url": "string or null",
  "other_links": ["string"],
  "skills": ["string"],
  "experience": [
    {
      "title": "string or null",
      "company": "string or null",
      "location": "string or null",
      "duration": "string or null",
      "achievements": ["string"],
      "technologies": ["string"]
    }
  ],
  "projects": [
    {
      "name": "string or null",
      "description": "string or null",
      "tech_stack": ["string"],
      "achievements": ["string"],
      "url": "string or null"
    }
  ],
  "education": [
    {
      "degree": "string or null",
      "institution": "string or null",
      "year": "string or null",
      "details": ["string"]
    }
  ],
  "certifications": ["string"],
  "awards": ["string"],
  "publications": ["string"],
  "languages": ["string"]
}

Rules:
- Extract total years of experience only when explicitly stated in the resume.
- Split skills into clean individual skill strings.
- Keep achievement bullets factual and concise.
- If a field is missing, use null or [].
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=images + [
                "Parse this scanned resume into the required JSON schema."
            ],
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
