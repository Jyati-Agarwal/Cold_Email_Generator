import os
import re
import json
import requests
from typing import Dict, List


class WebSearchError(Exception):
    pass


EMAIL_PATTERN = re.compile(
    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
)


def _extract_emails_from_sources(sources: List[Dict[str, str]]) -> List[str]:
    """Run email regex across all snippets in a source list."""
    combined = " ".join(s.get("snippet", "") for s in sources)
    return list(set(EMAIL_PATTERN.findall(combined)))


def research_company(
    company_name: str,
    role: str,
    has_email: bool,
) -> dict:
    """
    Research a company using Tavily search.

    Query A: Company intelligence (always runs).
    Query B: HR contact search (only runs if has_email is False).

    Returns structured sources and any emails found in snippets.
    """
    if not company_name:
        return {
            "company_summary_sources": [],
            "contact_sources": [],
            "found_emails": [],
            "search_ran": False,
        }

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise WebSearchError("TAVILY_API_KEY environment variable is not set")

    # FIX 2: API key goes in Authorization header, not the request body
    api_url = "https://api.tavily.com/search"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {tavily_api_key}",
    }

    result: dict = {
        "company_summary_sources": [],
        "contact_sources": [],
        "found_emails": [],
        "search_ran": True,
    }

    def execute_query(query: str, search_depth: str) -> List[Dict[str, str]]:
        payload = {
            "query": query,           # FIX 2: api_key removed from body
            "search_depth": search_depth,
            "max_results": 5,
        }
        try:
            # FIX 1: Corrected broken markdown-artifact URL call
            response = requests.post(api_url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            sources = []
            for item in data.get("results", []):
                # Prefer "content" (advanced search) over "snippet" (basic)
                snippet = item.get("content") or item.get("snippet", "")
                sources.append({
                    "title": item.get("title", ""),
                    "url":   item.get("url", ""),
                    "snippet": snippet[:400],
                })
            return sources

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "unknown"
            raise WebSearchError(
                f"Tavily returned HTTP {status} for query '{query}': {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            raise WebSearchError(
                f"Tavily request failed for query '{query}': {str(e)}"
            )

    # ── Query A: Company Intelligence (always runs) ──────────────────────────
    query_a = f'"{company_name}" company overview product technology stack culture'
    company_sources = execute_query(query_a, "basic")
    result["company_summary_sources"] = company_sources

    # FIX 3: Also scan Query A snippets for any incidental emails
    emails_from_a = _extract_emails_from_sources(company_sources)

    # ── Query B: HR Contact (only when no email was found upstream) ──────────
    emails_from_b: List[str] = []
    if not has_email:
        query_b = (
            f'"{company_name}" recruiter HR hiring manager contact email careers'
        )
        contact_sources = execute_query(query_b, "advanced")
        result["contact_sources"] = contact_sources
        emails_from_b = _extract_emails_from_sources(contact_sources)

    # Merge all found emails (deduplicated)
    result["found_emails"] = list(set(emails_from_a + emails_from_b))

    return result


if __name__ == "__main__":
    company = "Stripe"
    role = "Senior Software Engineer"

    print("=" * 50)
    print(f"Test 1: has_email=False (both queries run)")
    print("=" * 50)
    try:
        res = research_company(company_name=company, role=role, has_email=False)
        print(json.dumps(res, indent=2))
        assert res["search_ran"] is True
        assert isinstance(res["company_summary_sources"], list)
        assert isinstance(res["contact_sources"], list)
        assert isinstance(res["found_emails"], list)
        print("\n✓ All assertions passed")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print(f"Test 2: has_email=True (Query A only)")
    print("=" * 50)
    try:
        res = research_company(company_name=company, role=role, has_email=True)
        print(json.dumps(res, indent=2))
        assert res["search_ran"] is True
        assert res["contact_sources"] == []   # Query B must NOT have run
        print("\n✓ All assertions passed")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 50)
    print("Test 3: Empty company name (early return)")
    print("=" * 50)
    res = research_company(company_name="", role=role, has_email=False)
    assert res["search_ran"] is False
    print(json.dumps(res, indent=2))
    print("\n✓ All assertions passed")