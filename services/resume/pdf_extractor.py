import fitz  # PyMuPDF
from urllib.parse import urlparse

def extract_links_from_pdf(pdf_path: str) -> dict:
    """
    Extract hyperlink URIs from PDF annotation metadata.
    This is the ONLY reliable way to get embedded links from LaTeX resumes.
    Never relies on text content — reads /Annots layer directly.
    """
    links = {
        "github": None,
        "linkedin": None,
        "email": None,
        "other": [],
        "source": "annotation"
    }

    doc = None
    try:
        doc = fitz.open(pdf_path)
        all_uris = []

        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                if not uri:
                    continue
                uri = uri.strip()
                if not _is_valid_uri(uri):
                    continue
                all_uris.append(uri)

        for uri in all_uris:
            lower = uri.lower()
            if "github.com" in lower and links["github"] is None:
                links["github"] = _pick_best_github(
                    [u for u in all_uris if "github.com" in u.lower()]
                )
            elif "linkedin.com" in lower and links["linkedin"] is None:
                links["linkedin"] = uri
            elif uri.startswith("mailto:"):
                links["email"] = uri.replace("mailto:", "").strip()
            else:
                if uri not in links["other"]:
                    links["other"].append(uri)

    finally:
        if doc:
            doc.close()

    return links


def _is_valid_uri(uri: str) -> bool:
    if not uri or len(uri) < 6:
        return False
    try:
        parsed = urlparse(uri)
        return parsed.scheme in ("http", "https", "mailto") and bool(
            parsed.netloc or parsed.path
        )
    except Exception:
        return False


def _pick_best_github(github_uris: list[str]) -> str | None:
    if not github_uris:
        return None

    def score(url: str) -> int:
        # Profile URLs have exactly one path segment — prefer them
        path_parts = urlparse(url).path.strip("/").split("/")
        return 0 if len(path_parts) == 1 else 1

    return min(github_uris, key=score)


def extract_text_from_pdf(pdf_path: str) -> tuple[str, bool]:
    """
    Extract full text using page.get_text("text") across all pages.
    Returns (text, is_scanned).
    """
    full_text = ""
    doc = None
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            full_text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    finally:
        if doc:
            doc.close()

    is_scanned = len(full_text.strip()) < 100
    return full_text.strip(), is_scanned
