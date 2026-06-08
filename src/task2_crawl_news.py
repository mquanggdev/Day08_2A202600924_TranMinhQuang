"""Task 2 - crawl news articles into data/landing/news.

Fill ARTICLE_URLS with verified article URLs before running. The crawler writes
one JSON file per article with url/title/date_crawled/content_markdown.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS: list[str] = [
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://ngoisao.vnexpress.net/nam-than-lai-nga-nhikolai-dinh-bi-bat-4762594.html",
    "https://ngoisao.vnexpress.net/nhung-nghe-si-viet-nga-ngua-vi-ma-tuy-4816068.html",
    "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
    "https://vietnamnet.vn/van-hoa-giai-tri/toan-canh-vu-bat-tam-giam-long-nhat-va-son-ngoc-sk0008VN.html",
]


def _slug(text: str, fallback: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return value[:80] or fallback


async def crawl_article(url: str) -> dict:
    """Crawl one article using Crawl4AI when available, requests otherwise."""
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            metadata = getattr(result, "metadata", {}) or {}
            return {
                "url": url,
                "title": metadata.get("title", "Unknown"),
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": getattr(result, "markdown", "") or "",
            }
    except ImportError:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        title_match = re.search(r"<title[^>]*>(.*?)</title>", response.text, flags=re.I | re.S)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "Unknown"
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", response.text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": text,
        }


async def crawl_all():
    setup_directory()
    for i, url in enumerate(ARTICLE_URLS, 1):
        article = await crawl_article(url)
        filename = f"{i:02d}-{_slug(article.get('title', ''), 'article')}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Fill ARTICLE_URLS with at least 5 verified news URLs before crawling.")
    else:
        asyncio.run(crawl_all())
