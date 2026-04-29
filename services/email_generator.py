import os
import json
import re
from google import genai
from google.genai import types


class EmailGenerationError(Exception):
    pass


SYSTEM_INSTRUCTION = """
You are a senior professional writing a highly-tailored, direct cold email to get a job interview. You write like a human, not a cover letter bot.

Structure the email in this exact format:
1. Greeting — "Hi [Recipient Name]," (extract from the To: email if no name is available, use "Hi there,")
2. Opening hook — 1 sentence of genuine company-specific research
3. Explicit intent — Clearly state: "I'm writing to express my interest in the [Role Name] position"
4. Value proof — 2 sentences max, with specific metrics or scale (e.g. how many users, cities, teams). Pull numbers from the resume wherever possible
5. Why them specifically — 1 sentence connecting candidate's goals to this company's mission
6. Strong CTA — End with: "Would you be open to a 15-minute call this week?"
7. Sign-off — "Best, [Full Name]" followed by LinkedIn and GitHub URLs on separate lines

Subject line:
Make it compelling and specific. Format it like: "[Unique angle] — interested in [Role]". For example: "SIH '24 finalist interested in Founder's Office role at Shram"

Tone rules:
- Professional, confident, warm
- Under 200 words in the body
- Never use "I hope this email finds you well" or similar filler

Output format — return ONLY a valid JSON object, no markdown:
{
  "subject": "string",
  "body": "string"
}
"""


def _build_user_prompt(context: dict) -> str:
    """Build the structured user prompt from the clean context object."""
    project = context.get("strongest_project") or {}
    experience = context.get("strongest_experience") or {}
    skills = context.get("top_3_matching_skills") or []

    return f"""
Write a cold email using ONLY this data. Do not invent anything.

Candidate: {context.get("candidate_name", "Unknown")}
Role applying for: {context.get("role_applied_for", "Not specified")}
Company summary: "{context.get("company_summary") or "No company info available."}"
Top matching skills: {", ".join(skills) if skills else "Not specified"}
Best project: {project.get("name", "N/A")} — {project.get("description", "N/A")}
  Why relevant: {project.get("relevance", "N/A")}
Best experience achievement: {experience.get("key_achievement", "N/A")}
Portfolio: {context.get("portfolio_url") or "N/A"}
LinkedIn: {context.get("linkedin_url") or "N/A"}
GitHub: {context.get("github_url") or "N/A"}
"""


def _parse_subject_and_body(raw_text: str) -> tuple[str, str]:
    """
    Parse subject and body from Gemini response.
    Primary path: JSON object with 'subject' and 'body' keys.
    Fallback path: plain text with 'Subject: ...' line.
    """
    # Primary: JSON parse
    try:
        data = json.loads(raw_text)
        subject = (data.get("subject") or "").strip()
        body = (data.get("body") or "").strip()
        if subject and body:
            return subject, body
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: plain text with Subject: prefix
    lines = raw_text.strip().splitlines()
    subject = ""
    body_lines = []

    for line in lines:
        if re.match(r"^Subject\s*:", line, re.IGNORECASE):
            subject = re.sub(
                r"^Subject\s*:\s*", "", line, flags=re.IGNORECASE
            ).strip()
        else:
            body_lines.append(line)

    body = "\n".join(body_lines).strip()

    if not subject:
        raise EmailGenerationError(
            "Could not parse subject line from Gemini response."
        )

    return subject, body


def generate_email(context: dict) -> dict:
    """
    Generate a personalized cold email from a clean context object.

    Args:
        context: Output of context_builder.build_context()

    Returns:
        {"subject": str, "body": str, "to_email": str | None}
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EmailGenerationError(
            "GEMINI_API_KEY environment variable is not set"
        )
    client = genai.Client(api_key=api_key)
    user_prompt = _build_user_prompt(context)
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.65,
            ),
        )
        
        subject, body = _parse_subject_and_body(response.text)

        return {
            "subject": subject,
            "body": body,
            "to_email": context.get("hr_email") or None,
        }

    except EmailGenerationError:
        raise
    except json.JSONDecodeError as e:
        raise EmailGenerationError(
            f"Failed to parse email response as JSON: {str(e)}"
        )
    except Exception as e:
        raise EmailGenerationError(
            f"Failed to communicate with Gemini API: {str(e)}"
        )


if __name__ == "__main__":
    sample_context = {
        "candidate_name": "Jane Smith",
        "candidate_email": "jane@example.com",
        "top_3_matching_skills": ["Python", "FastAPI", "PostgreSQL"],
        "strongest_project": {
            "name": "AutoSearch",
            "description": "A semantic search engine built with FastAPI and pgvector",
            "relevance": (
                "Directly applies FastAPI and PostgreSQL"
                " — the core stack for this role."
            ),
        },
        "strongest_experience": {
            "company": "TechStartup Inc",
            "role": "Backend Engineer",
            "key_achievement": (
                "Reduced API p99 latency by 38% by introducing Redis caching"
            ),
        },
        "company_summary": (
            "Acme Corp builds developer tools for the cloud. "
            "Founded in 2018, they serve over 10,000 engineering teams worldwide."
        ),
        "company_industry": "Developer Tools",
        "hr_email": "jobs@acmecorp.com",
        "role_applied_for": "Senior Backend Engineer",
        "portfolio_url": "https://janesmith.dev",
        "github_url": "https://github.com/janesmith",
    }

    print("=" * 50)
    print("Test 1: Full context — standard flow")
    print("=" * 50)
    try:
        result = generate_email(sample_context)
        print(f"Subject : {result['subject']}")
        print(f"To      : {result['to_email']}")
        print(f"\nBody:\n{result['body']}")

        # FIX 2: .strip() before split to exclude leading/trailing whitespace
        # from inflating the word count
        word_count = len(result["body"].strip().split())
        print(f"\nWord count: {word_count}")
        assert word_count <= 150, f"FAIL: Body exceeds 150 words ({word_count})"
        assert result["to_email"] == "jobs@acmecorp.com"
        assert result["subject"] != ""
        # Verify first word is not "I"
        first_word = result["body"].strip().split()[0]
        assert first_word != "I", f"FAIL: Body starts with 'I'"
        print("\n✓ All assertions passed")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("Test 2: No company summary — graceful degradation")
    print("=" * 50)
    try:
        no_company_context = {**sample_context, "company_summary": None}
        result = generate_email(no_company_context)
        print(f"Subject : {result['subject']}")
        print(f"\nBody:\n{result['body']}")
        assert result["subject"] != ""
        assert result["body"] != ""
        print("\n✓ Graceful degradation passed")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("Test 3: No hr_email — to_email must be None")
    print("=" * 50)
    try:
        no_email_context = {**sample_context, "hr_email": None}
        result = generate_email(no_email_context)
        assert result["to_email"] is None
        print(f"to_email is None ✓")
        print("\n✓ to_email null-safety passed")
    except Exception as e:
        print(f"Error: {e}")