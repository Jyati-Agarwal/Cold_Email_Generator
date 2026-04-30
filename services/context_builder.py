import os
import json
from google import genai
from google.genai import types


class ContextBuildError(Exception):
    pass


def build_context(
    resume_data: dict,
    job_data: dict,
    search_data: dict,
) -> dict:
    """
    Synthesize raw data from all pipeline modules into a clean,
    LLM-ready context object using a single Gemini call.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ContextBuildError("GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)

    # -- Build hr_email priority list BEFORE sending to Gemini --
    # Priority: job_data emails > search found_emails > null
    job_emails: list[str] = job_data.get("emails") or []
    search_emails: list[str] = search_data.get("found_emails") or []
    all_candidate_emails = job_emails + [e for e in search_emails if e not in job_emails]
    hr_email_hint = all_candidate_emails[0] if all_candidate_emails else None

    # -- Compile company summary snippets from search data --
    company_snippets = [
        s.get("snippet", "")
        for s in (search_data.get("company_summary_sources") or [])
        if s.get("snippet")
    ]
    company_context_text = " | ".join(company_snippets[:3])  # top 3 snippets max

    system_instruction = """
You are a context synthesis engine. Given raw data from multiple sources,
produce a clean, factual summary. Never invent information. If a field has
no data, set it to null.

You will receive:
- Resume data (candidate info)
- Job data (company, role, requirements)
- Search snippets (company research from the web)
- A pre-resolved HR email (already prioritised correctly — use it as-is)

Return ONLY a valid JSON object matching this schema:
{
  "candidate_name": "string",
  "candidate_email": "string or null",
  "top_3_matching_skills": ["string"],
  "strongest_project": {
    "name": "string",
    "description": "string",
    "relevance": "string"
  },
  "strongest_experience": {
    "company": "string",
    "role": "string",
    "key_achievement": "string"
  },
  "company_summary": "string or null",
  "company_industry": "string or null",
  "hr_email": "string or null",
  "role_applied_for": "string or null",
  "portfolio_url": "string or null",
  "linkedin_url": "string or null",
  "github_url": "string or null"
}

Rules:
- "top_3_matching_skills" must ONLY include skills that appear in BOTH
  the resume skills list AND the job key_requirements. Do not invent matches.
  If fewer than 3 match, return what matches (even 0 or 1).
- "strongest_project" pick the project from resume most relevant to the role.
  If no projects are listed in resume, return null.
  "relevance" must explain WHY it is relevant to this specific role in one sentence.
- "strongest_experience" pick the experience with the most impressive content
  relative to the target role. "key_achievement" must be a concise summary of the impact.
- "company_summary" must be derived ONLY from the provided search snippets.
  Maximum 2 sentences. Do NOT hallucinate facts not present in the snippets.
- "hr_email" must be set to the pre-resolved value provided — do not change it.
- Return ONLY the JSON object. No markdown, no explanation.
"""

    user_prompt = f"""
Synthesise the following data into the required JSON context object.

--- RESUME DATA ---
{json.dumps(resume_data, indent=2)}

--- JOB DATA ---
{json.dumps(job_data, indent=2)}

--- COMPANY SEARCH SNIPPETS (from web, use for company_summary only) ---
{company_context_text if company_context_text else "No search data available."}

--- PRE-RESOLVED HR EMAIL (use this as hr_email, do not change) ---
{hr_email_hint if hr_email_hint else "null"}

--- TARGET ROLE ---
{job_data.get("role") or "Not specified"}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        context = json.loads(response.text)

        # -- Post-processing safety checks --
        # Enforce hr_email priority — override if Gemini ignores the hint
        if hr_email_hint and context.get("hr_email") != hr_email_hint:
            context["hr_email"] = hr_email_hint

        # Enforce top_3_matching_skills cap
        matching_skills: list[str] = context.get("top_3_matching_skills") or []
        context["top_3_matching_skills"] = matching_skills[:3]

        return context

    except json.JSONDecodeError as e:
        raise ContextBuildError(
            f"Failed to parse Gemini context response as JSON: {str(e)}"
        )
    except Exception as e:
        raise ContextBuildError(
            f"Failed to communicate with Gemini API: {str(e)}"
        )


if __name__ == "__main__":
    # Minimal test using synthetic data — no live APIs required
    sample_resume = {
        "name": "Jane Smith",
        "email": "jane@example.com",
        "phone": "+1-555-0100",
    "github_url": "https://github.com/janesmith",
    "linkedin_url": "https://linkedin.com/in/janesmith",
    "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "React"],
    "experience": [
      {
        "company": "TechStartup Inc",
        "title": "Backend Engineer",
        "duration": "2022–2024"
      }
    ],
        "projects": [
            {
                "name": "AutoSearch",
                "description": "A semantic search engine built with FastAPI and pgvector",
                "tech_stack": ["Python", "FastAPI", "PostgreSQL", "pgvector"],
                "url": "https://github.com/janesmith/autosearch"
            }
        ],
        "education": [
            {"institution": "MIT", "degree": "B.Sc Computer Science", "year": "2022"}
        ]
    }

    sample_job = {
        "company_name": "Acme Corp",
        "role": "Senior Backend Engineer",
        "emails": ["jobs@acmecorp.com"],
        "has_explicit_email": True,
        "apply_instructions": "Send resume to jobs@acmecorp.com",
        "key_requirements": ["Python", "FastAPI", "PostgreSQL", "Docker", "REST APIs"]
    }

    sample_search = {
        "company_summary_sources": [
            {
                "title": "Acme Corp Homepage",
                "url": "https://acmecorp.com",
                "snippet": "Acme Corp builds developer tools for the cloud. Founded in 2018, they serve over 10,000 engineering teams worldwide."
            }
        ],
        "contact_sources": [],
        "found_emails": [],
        "search_ran": True
    }

    print("=" * 50)
    print("Test: build_context with synthetic data")
    print("=" * 50)
    try:
        ctx = build_context(sample_resume, sample_job, sample_search)
        print(json.dumps(ctx, indent=2))
        assert ctx["candidate_name"] == "Jane Smith"
        assert ctx["hr_email"] == "jobs@acmecorp.com"
        assert len(ctx["top_3_matching_skills"]) <= 3
        assert ctx["company_summary"] is not None
        print("\n✓ All assertions passed")
    except Exception as e:
        print(f"Error: {e}")
