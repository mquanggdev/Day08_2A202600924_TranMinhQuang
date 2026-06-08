"""Task 3 - convert landing files into markdown under data/standardized."""

import json
import re
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _get_markitdown():
    try:
        from markitdown import MarkItDown

        return MarkItDown()
    except ImportError:
        return None


def _convert_pdf_with_pypdf(filepath: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(filepath))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"\n\n## Page {page_number}\n\n{text.strip()}")
    return "\n".join(pages).strip()


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def convert_legal_docs():
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not legal_dir.exists():
        return []

    md = _get_markitdown()
    outputs = []
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue
        try:
            if md is not None:
                text_content = md.convert(str(filepath)).text_content
            elif filepath.suffix.lower() == ".pdf":
                text_content = _convert_pdf_with_pypdf(filepath)
            else:
                raise RuntimeError(
                    "MarkItDown is required for DOC/DOCX conversion. "
                    "Install markitdown before converting this file."
                )
        except Exception:
            if filepath.suffix.lower() != ".pdf":
                raise
            # README path is MarkItDown. pypdf is a conservative fallback for
            # offline pytest runs when MarkItDown is unavailable or fails.
            text_content = _convert_pdf_with_pypdf(filepath)
        output_path = output_dir / f"{filepath.stem}.md"
        header = f"# {filepath.stem}\n\nSource: {filepath.name}\n\n"
        output_path.write_text(header + text_content, encoding="utf-8")
        outputs.append(output_path)
    return outputs


def convert_news_articles():
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not news_dir.exists():
        return []

    outputs = []
    for filepath in sorted(news_dir.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix == ".json":
            data = json.loads(filepath.read_text(encoding="utf-8"))
            title = data.get("title", "Unknown")
            header = (
                f"# {title}\n\n"
                f"Source: {data.get('url', 'N/A')}\n\n"
                f"Crawled: {data.get('date_crawled', data.get('crawl_date', 'N/A'))}\n\n"
            )
            body = data.get("content_markdown") or data.get("content") or ""
        elif suffix == ".html":
            header = f"# {filepath.stem}\n\nSource: {filepath.name}\n\n"
            body = _html_to_text(filepath.read_text(encoding="utf-8", errors="ignore"))
        elif suffix in (".md", ".txt"):
            header = f"# {filepath.stem}\n\nSource: {filepath.name}\n\n"
            body = filepath.read_text(encoding="utf-8", errors="ignore")
        else:
            continue

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + body, encoding="utf-8")
        outputs.append(output_path)
    return outputs


def convert_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    legal_outputs = convert_legal_docs()
    news_outputs = convert_news_articles()
    print(f"Converted {len(legal_outputs)} legal files and {len(news_outputs)} news files.")
    return legal_outputs + news_outputs


if __name__ == "__main__":
    convert_all()
