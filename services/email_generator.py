import os
import json
import re
from google import genai
from google.genai import types


class EmailGenerationError(Exception):
    pass


SYSTEM_INSTRUCTION = """
You are an expert job-application cold email writer. Write a direct,
specific, human email that feels like a sharp application note, not a cover
letter and not a marketing sequence.

What good job-application outreach does:
- Uses a clear, searchable subject line.
- Opens with a specific company/role hook only when factual context exists.
- Proves fit with 2-3 resume-backed facts, prioritizing actual company
  experience, role/title, YOE, internships, production impact, and metrics
  before personal projects.
- Makes one low-friction next-step ask.
- Keeps the message concise and easy to scan.

SUBJECT LINE RULE:
- Return the exact subject line supplied in EXPECTED SUBJECT.
- Do not add metrics, project names, emojis, or clickbait to the subject.

EMAIL BODY STRUCTURE:
1. Greeting:
   - Use "Hi [Recipient Name]," when recipient_name is present.
   - Otherwise use "Hi there,".
2. Opening:
   - If company_summary exists, write one specific sentence connecting the
     company or role context to the candidate.
   - If company_summary is null, skip the company hook and start with the role
     intent.
3. Intent:
   - State the candidate is applying for/interested in the role.
4. Proof:
   - If years_of_experience, current_or_recent_role, experience_summary, or
     company_experience exists, include it before mentioning projects.
   - Do not write a project-only email when company/professional experience is
     present in the context.
   - Include 2-3 concise proof points from experience_summary,
     current_or_recent_role, company_experience, strongest_experience,
     best_evidence, strongest_project, or application_angle.
   - Use bullets only if they improve scanability. Bullets must be short.
5. Skill match:
   - Mention 1-3 skills from top_3_matching_skills only when present.
6. CTA:
   - Ask for a quick conversation, interview consideration, or the best next
     step. Keep it professional and low-pressure.
7. Signature:
   - Include full name.
   - Include phone and email only when present.
   - Include at most three links: LinkedIn, GitHub profile, and a true
     portfolio/personal website if present. Do not dump every project URL.
   - Do not label a GitHub repository as Portfolio.

STRICT RULES:
- Body must be 120-180 words unless there is very little resume data.
- Never invent facts, metrics, company details, recipient names, or URLs.
- Never omit company experience/YOE when they are present in the context.
- Do not use generic filler such as "I hope this email finds you well",
  "I am passionate about", "leverage my skills", "synergy", or "dynamic team".
- Avoid weak filler like "I am confident my experience aligns well".
- Do not apologize, over-explain, or sound desperate.
- Return ONLY a valid JSON object, no markdown backticks, no explanation:
  {"subject": "string", "body": "string"}
"""


def _expected_subject(context: dict) -> str:
    role = context.get("role_applied_for") or "the role"
    name = context.get("candidate_name") or "Candidate"
    return f"Application for {role} - {name}"


def _context_links(context: dict) -> list[dict]:
    links = context.get("application_links") or []
    if links:
        allowed_labels = {"linkedin", "github", "portfolio"}
        filtered_links = [
            link
            for link in links
            if (link.get("label") or "").lower() in allowed_labels
        ]
        return filtered_links[:3]

    fallback_links = []
    for field, label in (
        ("linkedin_url", "LinkedIn"),
        ("github_url", "GitHub"),
        ("portfolio_url", "Portfolio"),
    ):
        url = context.get(field)
        if url:
            fallback_links.append({"label": label, "url": url})
    return fallback_links


def _build_user_prompt(context: dict) -> str:
    expected_subject = _expected_subject(context)
    safe_context = {
        "candidate_name": context.get("candidate_name"),
        "candidate_email": context.get("candidate_email"),
        "candidate_phone": context.get("candidate_phone"),
        "candidate_location": context.get("candidate_location"),
        "candidate_headline": context.get("candidate_headline"),
        "years_of_experience": context.get("years_of_experience"),
        "current_or_recent_role": context.get("current_or_recent_role"),
        "experience_summary": context.get("experience_summary"),
        "company_experience": context.get("company_experience") or [],
        "recipient_name": context.get("recipient_name"),
        "role_applied_for": context.get("role_applied_for"),
        "company_summary": context.get("company_summary"),
        "company_industry": context.get("company_industry"),
        "top_3_matching_skills": context.get("top_3_matching_skills") or [],
        "strongest_project": context.get("strongest_project"),
        "strongest_experience": context.get("strongest_experience"),
        "best_evidence": context.get("best_evidence") or [],
        "application_angle": context.get("application_angle"),
        "application_links": _context_links(context),
    }

    return f"""
Write a cold application email using ONLY the data provided below.
Do not invent any information not present here.

EXPECTED SUBJECT:
{expected_subject}

EMAIL CONTEXT:
{json.dumps(safe_context, indent=2)}
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
        if body:
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

    if not body:
        raise EmailGenerationError(
            "Could not parse email body from Gemini response."
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
        
        _subject, body = _parse_subject_and_body(response.text)
        expected_subject = _expected_subject(context)

        return {
            "subject": expected_subject,
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
