"""
Crawler LIGHT – indexează DOAR articolele noi.
Ideal pentru cron job zilnic.
"""

import os
import time
import uuid
import requests
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    VectorParams, Distance, PointStruct
)
from qdrant_client.http.exceptions import ResponseHandlingException

BASE_URL = "https://ordinesaudezordine.com/"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30.0
)

BAD_LINK_PARTS = [
    "facebook.com", "twitter.com", "linkedin.com", "pinterest",
    "utm_", "share", "login", "wp-login", "password", "checkpoint",
    "r.php", "redirect", "wp-json", "mailto:", "tel:"
]


def fetch_latest_articles():
    """Returnează link-urile articolelor din pagina principală a blogului."""
    print("[INFO] Pregătesc lista articolelor...")
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        href, _ = urldefrag(href)

        if any(bad in href.lower() for bad in BAD_LINK_PARTS):
            continue

        if href.startswith(BASE_URL) and len(href.split("/")) > 4:
            links.append(href)

    return list(set(links))


def article_already_indexed(url: str) -> bool:
    """Verifică dacă articolul există deja în Qdrant."""
    scroll = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="url",
                    match=MatchValue(value=url)
                )
            ]
        ),
        limit=1
    )
    return len(scroll[0]) > 0


def safe_upsert(points_batch):
    """Upsert stabil + retry."""
    for retry in range(5):
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            return True
        except ResponseHandlingException:
            print(f"[WARN] Timeout Qdrant, retry {retry+1}/5…")
            time.sleep(1 + retry)
    print("[FATAL] Qdrant nu răspunde.")
    return False


def process_article(url: str):
    print(f"\n[CRAWL] {url}")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except:
        print(f"[SKIP] Nu pot accesa {url}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "svg", "noscript", "header", "footer"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    if len(text.split()) < 30:
        print("[SKIP] Prea puțin text.")
        return

    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else url

    chunks = chunk_text(text)
    vectors = embed_texts(chunks)

    points = []
    for vec, chunk in zip(vectors, chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "url": url,
                    "title": title,
                    "text": chunk
                }
            )
        )

    print(f"[INFO] {len(points)} bucăți de text → upload în Qdrant...")
    batch_upload(points, batch_size=3)


def chunk_text(text, max_tokens=350):
    words = text.split()
    chunks, chunk = [], []

    for w in words:
        chunk.append(w)
        if len(chunk) >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def embed_texts(texts):
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def batch_upload(points, batch_size=3):
    while points:
        batch = points[:batch_size]
        if safe_upsert(batch):
            print(f"[UPSERT] {len(batch)} puncte trimise.")
        points = points[batch_size:]


def main():
    print("=== START CRAWLER LIGHT ===")
    articles = fetch_latest_articles()
    print(f"[INFO] Am găsit {len(articles)} articole în homepage.")

    new_articles = []

    for url in articles:
        if not article_already_indexed(url):
            new_articles.append(url)

    if not new_articles:
        print("[INFO] Nu există articole noi.")
        return

    print(f"[INFO] Articole noi găsite: {len(new_articles)}")

    for article_url in new_articles:
        process_article(article_url)

    print("=== GATA LIGHT INDEX ===")


if __name__ == "__main__":
    main()
