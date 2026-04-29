import json
import io
import os
from pypdf import PdfReader
from google import genai
from google.genai import types


class ParseError(Exception):
    pass


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        raise ParseError(f"Failed to read PDF: {str(e)}")

    if len(text.strip()) < 100:
        raise ParseError(
            "PDF text extraction failed — file may be image-based"
        )

    return text.strip()


def parse_resume(pdf_bytes: bytes) -> dict:
    """
    Parse resume from PDF bytes and return structured data using Gemini.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ParseError("GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    raw_text = extract_text_from_pdf(pdf_bytes)

    system_instruction = """
You are an expert resume parser. Extract the candidate's information from
the provided resume text and return it strictly as a valid JSON object
matching this schema:

{
  "name": "string",
  "email": "string or null",
  "phone": "string or null",
  "github": "string (URL) or null",
  "linkedin": "string (URL) or null",
  "portfolio": "string (URL) or null",
  "skills": ["string"],
  "experience": [
    {
      "company": "string",
      "role": "string",
      "duration": "string",
      "achievements": ["string"]
    }
  ],
  "projects": [
    {
      "name": "string",
      "description": "string",
      "tech_stack": ["string"],
      "url": "string or null"
    }
  ],
  "education": [
    {
      "institution": "string",
      "degree": "string",
      "year": "string"
    }
  ]
}

Rules:
- "achievements" must be specific bullet-point accomplishments (e.g.,
  "Reduced API latency by 40% by introducing Redis caching"), not generic
  job descriptions.
- If a field is not present in the resume, use null for strings and []
  for lists.
- Return ONLY the JSON object. No markdown, no explanation.
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=raw_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        return json.loads(response.text)

    except json.JSONDecodeError as e:
        raise ParseError(
            f"Failed to parse Gemini response as JSON: {str(e)}"
        )
    except Exception as e:
        raise ParseError(
            f"Failed to communicate with Gemini API: {str(e)}"
        )


if __name__ == "__main__":
    test_pdf_path = "test_resume.pdf"

    if os.path.exists(test_pdf_path):
        with open(test_pdf_path, "rb") as f:
            try:
                # FIX 3: Corrected broken f.read() call (was a markdown artifact)
                result = parse_resume(f.read())
                print(json.dumps(result, indent=2))
            except ParseError as e:
                print(f"ParseError: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
    else:
        print(f"Please place a '{test_pdf_path}' file here to run the test.")