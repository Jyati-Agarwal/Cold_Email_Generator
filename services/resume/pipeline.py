import os
from typing import Dict, Any
from .pdf_extractor import (
    extract_links_from_pdf, 
    extract_text_from_pdf
)
from .text_fallback import extract_links_from_text
from .gemini_parser import (
    parse_with_gemini_text, 
    parse_with_gemini_vision
)

def parse_resume(pdf_path: str, client) -> dict:
    # Track 1: Extract links from PDF annotations (ground truth)
    extracted_links = extract_links_from_pdf(pdf_path)

    # Track 2: Extract text, detect scanned PDFs
    text, is_scanned = extract_text_from_pdf(pdf_path)

    # Track 3: LLM parses text content only — NOT links
    if is_scanned:
        parsed = parse_with_gemini_vision(pdf_path, client)
    else:
        parsed = parse_with_gemini_text(text, client)

    # Track 4: Regex fallback if annotations returned nothing
    link_source = "annotation"
    if not extracted_links["github"] and not extracted_links["linkedin"]:
        fallback = extract_links_from_text(text)
        if fallback.get("github") or fallback.get("linkedin"):
            extracted_links.update(fallback)
            link_source = "text_regex"
        else:
            link_source = "not_found"

    # Merge — PyMuPDF always wins on link fields, never trust LLM for URLs
    parsed["github_url"] = extracted_links.get("github") or parsed.get("github_url")
    parsed["linkedin_url"] = extracted_links.get("linkedin") or parsed.get("linkedin_url")
    parsed["email"] = extracted_links.get("email") or parsed.get("email")
    parsed["link_source"] = link_source
    parsed["is_scanned"] = is_scanned

    return parsed
