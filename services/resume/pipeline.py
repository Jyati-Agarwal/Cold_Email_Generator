from .pdf_extractor import (
    extract_links_from_pdf, 
    extract_text_from_pdf
)
from urllib.parse import urlparse
from .text_fallback import extract_links_from_text
from .gemini_parser import (
    parse_with_gemini_text, 
    parse_with_gemini_vision
)


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


def _as_list(value) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _link_source(annotation_links: dict, text_links: dict) -> str:
    annotation_has_links = any(
        annotation_links.get(key)
        for key in ("github", "linkedin", "portfolio", "email", "all_links")
    )
    text_has_links = any(
        text_links.get(key)
        for key in ("github", "linkedin", "portfolio", "email", "all_links")
    )
    if annotation_has_links and text_has_links:
        return "annotation+text_regex"
    if annotation_has_links:
        return "annotation"
    if text_has_links:
        return "text_regex"
    return "not_found"


def _valid_portfolio_url(url):
    if not url:
        return None

    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if "github.com" in netloc:
        path_parts = [part for part in parsed.path.strip("/").split("/") if part]
        # A GitHub profile belongs in github_url; a repo should stay in links,
        # not be mislabeled as a personal portfolio.
        return None if len(path_parts) >= 1 else url
    return url


def parse_resume(pdf_path: str, client) -> dict:
    # Track 1: Extract links from PDF annotations (ground truth)
    extracted_links = extract_links_from_pdf(pdf_path)

    # Track 2: Extract text, detect scanned PDFs
    text, is_scanned = extract_text_from_pdf(pdf_path)

    # Track 3: LLM parses structured resume evidence; deterministic extraction
    # still wins for contact links after parsing.
    if is_scanned:
        parsed = parse_with_gemini_vision(pdf_path, client)
    else:
        parsed = parse_with_gemini_text(text, client)

    # Track 4: Regex fallback from visible text. This runs even when annotation
    # links exist because resumes often expose phone numbers, portfolio URLs, or
    # profile text that are not embedded as hyperlink annotations.
    text_links = extract_links_from_text(text)

    # Merge — annotation links win when present, text regex fills visible data,
    # and Gemini Vision can still contribute exact visible URLs for scanned PDFs.
    parsed["github_url"] = (
        extracted_links.get("github")
        or text_links.get("github")
        or parsed.get("github_url")
    )
    parsed["linkedin_url"] = (
        extracted_links.get("linkedin")
        or text_links.get("linkedin")
        or parsed.get("linkedin_url")
    )
    parsed["portfolio_url"] = _valid_portfolio_url(
        extracted_links.get("portfolio")
        or text_links.get("portfolio")
        or parsed.get("portfolio_url")
    )
    parsed["email"] = (
        extracted_links.get("email")
        or text_links.get("email")
        or parsed.get("email")
    )
    parsed["phone"] = text_links.get("phone") or parsed.get("phone")

    other_links = _dedupe_preserve_order(
        _as_list(extracted_links.get("other"))
        + _as_list(text_links.get("other_links"))
        + _as_list(parsed.get("other_links"))
    )
    primary_links = [
        parsed.get("linkedin_url"),
        parsed.get("github_url"),
        parsed.get("portfolio_url"),
    ]
    parsed["other_links"] = [
        link for link in other_links if link not in set(filter(None, primary_links))
    ]
    parsed["links"] = _dedupe_preserve_order(
        [link for link in primary_links if link] + parsed["other_links"]
    )
    parsed["emails"] = _dedupe_preserve_order(
        _as_list(text_links.get("emails")) + _as_list(parsed.get("email"))
    )
    parsed["phones"] = _dedupe_preserve_order(
        _as_list(text_links.get("phones")) + _as_list(parsed.get("phone"))
    )
    parsed["link_source"] = _link_source(extracted_links, text_links)
    parsed["is_scanned"] = is_scanned

    return parsed
