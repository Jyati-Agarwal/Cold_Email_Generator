import os
import json
import re
from google import genai
from google.genai import types


class EmailGenerationError(Exception):
    pass


SYSTEM_INSTRUCTION = """
You are writing a professional cold job application email on behalf of a candidate.
Your goal is to write something direct, specific, and human — not a cover letter.

SUBJECT LINE RULE — follow this exactly, no exceptions:
Format: "Application for [Role Name] – [Candidate Full Name]"
Example: "Application for Junior Data Analyst – Jyati Agarwal"
Do NOT use project names, metrics, achievements, or creative angles in the subject line.
Do NOT deviate from this format under any circumstance.

EMAIL BODY STRUCTURE — follow this order exactly:

1. Greeting: "Hi there," — always use this if no recipient name is available

2. Opening (1 sentence): One genuine, specific observation about the company
   derived ONLY from the company_summary field. If company_summary is null,
   skip this sentence entirely — do not invent or hallucinate company details.

3. Intent (1 sentence): "I'm writing to express my interest in the [Role Name] position."

4. Value proof (2 sentences maximum):
   - Sentence 1: Mention the strongest_project name and what it does technically.
     Include a real metric only if one is present in the data — do not invent numbers.
   - Sentence 2: Mention the strongest_experience key_achievement.
     Again, only use metrics that are explicitly provided.

5. Skill alignment (1 sentence): Connect 1–2 skills from top_3_matching_skills
   to the role's actual requirements. Be specific — name the skills explicitly.

6. End with a good professional closing line based on the context of the resume and job description, for example - Would you be open to a 15-minute call this week?

7. Sign-off:
   "Best,
   [Full Name]"
   Then on separate lines, include LinkedIn and GitHub URLs only if they are
   non-null in the data. If null, omit them entirely — do not write placeholder text.

STRICT RULES:
- Total body word count must be under 200 words
- Never start any sentence with "I" as the very first word of the email body
- Never use phrases like: "I hope this email finds you well", "I am excited to",
  "I am passionate about", "leverage my skills", "synergy", "I wanted to reach out"
- Never invent facts, metrics, or company details not present in the input data
- Never include LinkedIn/GitHub lines if those URLs are null
- Return ONLY a valid JSON object, no markdown backticks, no explanation:
  {"subject": "string", "body": "string"}
"""


def _build_user_prompt(context: dict) -> str:
    project = context.get("strongest_project") or {}
    experience = context.get("strongest_experience") or {}
    skills = context.get("top_3_matching_skills") or []
    github = context.get("github_url")
    linkedin = context.get("linkedin_url")

    return f"""
Write a cold application email using ONLY the data provided below.
Do not invent any information not present here.

CANDIDATE FULL NAME (use exactly as-is in subject line and sign-off): {context.get("candidate_name", "Unknown")}
ROLE APPLYING FOR: {context.get("role_applied_for", "Not specified")}
COMPANY SUMMARY (use only for opening hook, max 1 sentence): "{context.get("company_summary") or "null — skip the opening hook entirely"}"
TOP MATCHING SKILLS: {", ".join(skills) if skills else "none identified"}
BEST PROJECT NAME: {project.get("name", "N/A")}
BEST PROJECT DESCRIPTION: {project.get("description", "N/A")}
BEST PROJECT RELEVANCE TO ROLE: {project.get("relevance", "N/A")}
BEST EXPERIENCE ACHIEVEMENT: {experience.get("key_achievement", "N/A")}
LINKEDIN URL: {linkedin if linkedin else "null — do not include in email"}
GITHUB URL: {github if github else "null — do not include in email"}

REMINDER: Subject line must be exactly: "Application for {context.get("role_applied_for", "the role")} – {context.get("candidate_name", "Candidate")}"
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