from __future__ import annotations

import json
import re
import os
from google import genai
from google.genai import types

class JobParseError(Exception):
    pass


def parse_job_description(
    text: str | None = None,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> dict:
    """
    Parse a job description from text or image using Gemini.
    Returns structured data including company, role, emails, and requirements.
    """
    if not text and not image_bytes:
        raise JobParseError("Must provide either 'text' or 'image_bytes'")

    if image_bytes and not image_mime_type:
        raise JobParseError(
            "Must provide 'image_mime_type' when 'image_bytes' is provided"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise JobParseError("GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # STEP 1: Regex email extraction on raw text (before any LLM call)
    regex_emails: list[str] = []
    if text:
        email_pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
        regex_emails = list(set(re.findall(email_pattern, text)))

    system_instruction = """
You are an expert job description parser. Extract structured hiring context
from the provided job description or screenshot and return it strictly as a
valid JSON object matching this schema:

{
  "company_name": "string or null",
  "role": "string or null",
  "location": "string or null",
  "employment_type": "string or null",
  "seniority": "string or null",
  "emails": ["string"],
  "has_explicit_email": true or false,
  "apply_instructions": "string or null",
  "key_requirements": ["string"],
  "responsibilities": ["string"],
  "nice_to_have": ["string"],
  "company_context": "string or null",
  "job_url": "string or null",
  "recipient_name": "string or null"
}

Rules:
- "key_requirements" must list the top 5 most important skills or
  qualifications from the job description. Return exactly 5 if possible.
- "responsibilities" must list the top 3 role responsibilities if present.
- "nice_to_have" must include optional/preferred skills only if clearly present.
- "emails" should include any contact or application email addresses
  visible in the description.
- "recipient_name" should be a recruiter/hiring manager/contact name only if
  explicitly visible.
- If a field is not present, use null for strings and [] for lists.
- Return ONLY the JSON object. No markdown, no explanation.
"""

    try:
        # STEP 2: Build contents
        if image_bytes:
            contents = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image_mime_type,
                ),
                text if text else "Extract all job details from this image.",
            ]
        else:
            contents = text

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        result = json.loads(response.text)

        # STEP 3: Merge regex emails with LLM emails (deduplicate)
        llm_emails: list[str] = result.get("emails") or []
        all_emails = list(set(regex_emails + llm_emails))
        result["emails"] = all_emails
        result["has_explicit_email"] = len(all_emails) > 0

        # FIX 2: Enforce max 5 key_requirements — slice after parsing
        requirements: list[str] = result.get("key_requirements") or []
        result["key_requirements"] = requirements[:5]
        result["responsibilities"] = (result.get("responsibilities") or [])[:3]

        return result

    except json.JSONDecodeError as e:
        raise JobParseError(
            f"Failed to parse Gemini response as JSON: {str(e)}"
        )
    except Exception as e:
        raise JobParseError(
            f"Failed to communicate with Gemini API: {str(e)}"
        )


if __name__ == "__main__":
    # Test 1: Text input with two explicit emails
    test_text = """
    We are looking for a Senior Software Engineer to join Acme Corp.
    You will be working with Python, FastAPI, and React.
    Please apply by sending your resume to jobs@acmecorp.com or careers@acmecorp.com.
    Requirements:
    - 5+ years of Python experience
    - Experience with LLMs and vector databases
    - Familiarity with cloud platforms (AWS/GCP)
    - Strong communication skills
    - Experience with REST API design
    """

    print("=" * 50)
    print("Test 1: Text input")
    print("=" * 50)
    try:
        result = parse_job_description(text=test_text)
        print(json.dumps(result, indent=2))
        assert result["has_explicit_email"] is True
        assert "jobs@acmecorp.com" in result["emails"]
        assert len(result["key_requirements"]) <= 5
        print("\n✓ All assertions passed")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: No email in description
    test_text_no_email = """
    Backend Engineer at StartupXYZ.
    We use Go, Kubernetes, and PostgreSQL.
    Apply through our website at startupxyz.com/careers.
    """

    print("\n" + "=" * 50)
    print("Test 2: No email in description")
    print("=" * 50)
    try:
        result = parse_job_description(text=test_text_no_email)
        print(json.dumps(result, indent=2))
        assert result["has_explicit_email"] is False
        assert result["emails"] == []
        print("\n✓ All assertions passed")
    except Exception as e:
        print(f"Error: {e}")
