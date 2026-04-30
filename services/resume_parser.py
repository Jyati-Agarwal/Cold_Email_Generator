import os
import tempfile
from google import genai
from .resume.pipeline import parse_resume as execute_pipeline

class ParseError(Exception):
    pass

def parse_resume(pdf_bytes: bytes) -> dict:
    """
    Wrapper for the new multi-track resume parsing architecture.
    Integrates PyMuPDF link extraction with Gemini text/vision parsing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ParseError("GEMINI_API_KEY environment variable is not set")
    
    client = genai.Client(api_key=api_key)
    
    # Write bytes to a temporary file because PyMuPDF (fitz) works best with file paths
    # and the pipeline is designed to take a path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
        
    try:
        result = execute_pipeline(tmp_path, client)
        return result
    except Exception as e:
        raise ParseError(f"Resume parsing failed: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass