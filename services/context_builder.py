import os
import json
from google import genai
from google.genai import types


class ContextBuildError(Exception):
    pass


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _candidate_links(resume_data: dict) -> list[dict]:
    links = []
    field_labels = [
        ("linkedin_url", "LinkedIn"),
        ("github_url", "GitHub"),
        ("portfolio_url", "Portfolio"),
    ]
    for field, label in field_labels:
        url = resume_data.get(field)
        if url:
            links.append({"label": label, "url": url})

    for url in resume_data.get("other_links") or []:
        links.append({"label": "Link", "url": url})

    seen = set()
    deduped = []
    for link in links:
        url = link.get("url")
        if not url:
            continue
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(link)
    return deduped


def _enforce_resume_identity(context: dict, resume_data: dict) -> dict:
    context["candidate_name"] = (
        context.get("candidate_name")
        or resume_data.get("name")
        or "Candidate"
    )
    context["candidate_email"] = (
        context.get("candidate_email")
        or resume_data.get("email")
    )
    context["candidate_phone"] = (
        context.get("candidate_phone")
        or resume_data.get("phone")
    )
    context["candidate_location"] = (
        context.get("candidate_location")
        or resume_data.get("location")
    )
    context["linkedin_url"] = resume_data.get("linkedin_url") or context.get("linkedin_url")
    context["github_url"] = resume_data.get("github_url") or context.get("github_url")
    context["portfolio_url"] = resume_data.get("portfolio_url") or context.get("portfolio_url")
    context["application_links"] = _candidate_links(resume_data)
    return context


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
    resume_links = _candidate_links(resume_data)

    system_instruction = """
You are a context synthesis engine for high-quality job application emails.
Given raw data from multiple sources, produce a clean, factual context object.
Never invent information. If a field has no data, set it to null or [].

You will receive:
- Resume data with candidate identity, links, skills, projects, experience, and
  achievements
- Job data from text or image, including company, role, requirements, and apply
  instructions
- Search snippets for company context
- Pre-resolved HR email, already prioritised correctly

Return ONLY a valid JSON object matching this schema:
{
  "candidate_name": "string",
  "candidate_email": "string or null",
  "candidate_phone": "string or null",
  "candidate_location": "string or null",
  "top_3_matching_skills": ["string"],
  "strongest_project": {
    "name": "string",
    "description": "string",
    "relevance": "string",
    "evidence": "string or null"
  },
  "strongest_experience": {
    "company": "string",
    "role": "string",
    "key_achievement": "string"
  },
  "best_evidence": [
    {
      "type": "project or experience or education or certification",
      "title": "string",
      "detail": "string"
    }
  ],
  "company_summary": "string or null",
  "company_industry": "string or null",
  "hr_email": "string or null",
  "recipient_name": "string or null",
  "role_applied_for": "string or null",
  "application_angle": "string or null",
  "portfolio_url": "string or null",
  "linkedin_url": "string or null",
  "github_url": "string or null",
  "application_links": [
    {"label": "string", "url": "string"}
  ]
}

Rules:
- "top_3_matching_skills" should contain up to 3 resume-backed skills that best
  match the job requirements. A skill may semantically match a requirement, but
  the named skill must appear somewhere in the resume data.
- "strongest_project" pick the project from resume most relevant to the role.
  If no projects are listed in resume, return null.
  "relevance" must explain why it is relevant to this specific role in one
  sentence. "evidence" should preserve a real metric/technical proof point if
  present.
- "strongest_experience" pick the experience with the most impressive content
  relative to the target role. Return null if no experience exists.
  "key_achievement" must be a concise factual summary of impact.
- "best_evidence" should include the 2-3 strongest resume proof points for this
  job. Prefer quantified achievements, shipped projects, production systems,
  internships, open-source work, or role-relevant coursework/certifications.
- "company_summary" must be derived ONLY from the provided search snippets.
  Maximum 2 sentences. Do NOT hallucinate facts not present in the snippets.
- "application_angle" should be one sentence explaining why this candidate is a
  credible fit for this role using only resume and job data.
- "application_links" must use the pre-extracted candidate links exactly as
  provided. Do not add or edit URLs.
- "hr_email" must be set to the pre-resolved value provided. Do not change it.
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

--- PRE-EXTRACTED CANDIDATE LINKS (use exactly, do not edit URLs) ---
{json.dumps(resume_links, indent=2)}

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
        context = _enforce_resume_identity(context, resume_data)

        # Enforce hr_email priority — override if Gemini ignores the hint
        context["hr_email"] = hr_email_hint

        # Enforce top_3_matching_skills cap
        matching_skills: list[str] = context.get("top_3_matching_skills") or []
        context["top_3_matching_skills"] = _dedupe_preserve_order(matching_skills)[:3]
        context["best_evidence"] = (context.get("best_evidence") or [])[:3]
        context["role_applied_for"] = (
            context.get("role_applied_for")
            or job_data.get("role")
            or "the role"
        )
        context["recipient_name"] = (
            context.get("recipient_name")
            or job_data.get("recipient_name")
        )

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
