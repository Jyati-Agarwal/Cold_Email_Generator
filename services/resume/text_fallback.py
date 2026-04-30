from __future__ import annotations

import re
from urllib.parse import urlparse


EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])"
)

PHONE_PATTERN = re.compile(
    r"""
    (?<!\w)
    (?:\+?\d{1,3}[\s.\-]?)?
    (?:\(?\d{2,4}\)?[\s.\-]?){2,4}
    \d{2,4}
    (?!\w)
    """,
    re.VERBOSE,
)

URL_PATTERN = re.compile(
    r"""
    (?ix)
    (?:
        https?://[^\s<>()\[\]{}"']+
        |
        www\.[^\s<>()\[\]{}"']+
        |
        (?:github\.com|linkedin\.com/(?:in|pub)|gitlab\.com|bitbucket\.org|
           behance\.net|dribbble\.com|medium\.com|dev\.to|kaggle\.com|
           leetcode\.com|hackerrank\.com)/[^\s<>()\[\]{}"']+
        |
        (?<!@)\b[A-Za-z0-9][A-Za-z0-9-]{1,63}
        (?:\.[A-Za-z]{2,})+
        (?!@)
        (?:/[^\s<>()\[\]{}"']*)?
    )
    """,
    re.VERBOSE,
)

TRAILING_PUNCTUATION = ".,;:!?)\"]}'"
FALSE_POSITIVE_DOMAINS = {
    "asp.net",
    "ado.net",
    "vb.net",
    "next.js",
    "node.js",
    "react.js",
    "vue.js",
    "three.js",
    "d3.js",
    "chart.js",
    "express.js",
    "socket.io",
}


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _normalize_url(raw_url: str) -> str | None:
    url = raw_url.strip().rstrip(TRAILING_PUNCTUATION)
    if not url:
        return None

    if url.lower().startswith("mailto:"):
        return None

    if url.lower().startswith("www."):
        return f"https://{url}"

    if not re.match(r"(?i)^https?://", url):
        url = f"https://{url}"

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None

    return url


def _extract_urls(text: str) -> list[str]:
    text_without_emails = EMAIL_PATTERN.sub(" ", text or "")
    urls = []
    for match in URL_PATTERN.finditer(text_without_emails):
        url = _normalize_url(match.group(0))
        if url and not _looks_like_false_positive_url(url):
            urls.append(url)
    return _dedupe_preserve_order(urls)


def _looks_like_false_positive_url(url: str) -> bool:
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain in FALSE_POSITIVE_DOMAINS:
        return True
    if domain.endswith(".js") and not parsed.path.strip("/"):
        return True
    return False


def _extract_phones(text: str) -> list[str]:
    phones = []
    for match in PHONE_PATTERN.finditer(text or ""):
        phone = match.group(0).strip()
        digits = re.sub(r"\D", "", phone)
        if 10 <= len(digits) <= 15:
            phones.append(phone)
    return _dedupe_preserve_order(phones)


def _is_github_profile(url: str) -> bool:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.strip("/").split("/") if part]
    return "github.com" in parsed.netloc.lower() and len(path_parts) == 1


def _is_linkedin_profile(url: str) -> bool:
    parsed = urlparse(url)
    return "linkedin.com" in parsed.netloc.lower() and (
        parsed.path.lower().startswith("/in/")
        or parsed.path.lower().startswith("/pub/")
    )


def _pick_best_github(urls: list[str]) -> str | None:
    github_urls = [url for url in urls if "github.com" in urlparse(url).netloc.lower()]
    if not github_urls:
        return None

    profile_urls = [url for url in github_urls if _is_github_profile(url)]
    return (profile_urls or github_urls)[0]


def _pick_best_linkedin(urls: list[str]) -> str | None:
    linkedin_urls = [url for url in urls if "linkedin.com" in urlparse(url).netloc.lower()]
    if not linkedin_urls:
        return None

    profile_urls = [url for url in linkedin_urls if _is_linkedin_profile(url)]
    return (profile_urls or linkedin_urls)[0]


def _pick_best_portfolio(urls: list[str]) -> str | None:
    blocked_domains = {
        "github.com",
        "www.github.com",
        "linkedin.com",
        "www.linkedin.com",
        "gitlab.com",
        "bitbucket.org",
        "leetcode.com",
        "hackerrank.com",
        "kaggle.com",
    }
    for url in urls:
        netloc = urlparse(url).netloc.lower()
        if netloc not in blocked_domains:
            return url
    return None


def extract_links_from_text(text: str) -> dict:
    """
    Regex fallback for text extracted from resumes.

    This captures visible contact details and profile links that are not present
    in the PDF annotation layer. Annotation links still take precedence in the
    resume pipeline because they preserve embedded hyperlink targets.
    """
    emails = _dedupe_preserve_order(EMAIL_PATTERN.findall(text or ""))
    phones = _extract_phones(text or "")
    urls = _extract_urls(text or "")

    github = _pick_best_github(urls)
    linkedin = _pick_best_linkedin(urls)
    portfolio = _pick_best_portfolio(urls)

    primary_links = {link for link in (github, linkedin, portfolio) if link}
    other_links = [url for url in urls if url not in primary_links]

    return {
        "github": github,
        "linkedin": linkedin,
        "portfolio": portfolio,
        "email": emails[0] if emails else None,
        "emails": emails,
        "phone": phones[0] if phones else None,
        "phones": phones,
        "all_links": urls,
        "other_links": other_links,
        "source": "text_regex",
    }
