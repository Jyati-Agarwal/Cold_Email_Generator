import re

def extract_links_from_text(text: str) -> dict:
    """
    Regex fallback for plain-text resumes with no annotation layer.
    Lower confidence than annotation extraction — always flagged as text_regex.
    """
    github = re.search(r'github\.com/[\w\-]+', text, re.IGNORECASE)
    linkedin = re.search(r'linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
    email = re.search(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', text)

    return {
        "github": f"https://{github.group()}" if github else None,
        "linkedin": f"https://{linkedin.group()}" if linkedin else None,
        "email": email.group() if email else None,
        "source": "text_regex"
    }
