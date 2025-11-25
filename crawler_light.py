"""
CRAWLER LIGHT – Indexează DOAR articolele noi.
Filtrează media, HTML-only, rapid pentru cron job.
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

MEDIA_EXT = [
    ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".mp4", ".mov", ".avi", ".mp3", ".webm",
    ".pdf", ".zip", ".rar", ".7z",
    ".doc", ".docx"
]


def fetch_latest_articles():
    """Returnează link-urile articolelor din homepage."""
    print("[INFO] Preiau articolele din homepage...")
    resp = requests.get(BASE_URL, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(BASE_URL, a["href"])
        href, _ = urldefrag(href)

        if any(href.lower().endswith(ext) for ext in MEDIA_EXT):
            continue

        if href.startswith(BASE_URL) and len(href.split("/")) > 4:
            links.append(href)

    return list(set(links))


def article_already_indexed(url: str) -> bool:
    """Verifică dacă articolul este deja în Qdrant."""
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


def process_article(url: str):
    print(f"\n[CRAWL] {url}")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception:
        print("[SKIP] Nu pot accesa articolul.")
        return

    # Acceptăm doar HTML
    if "text/html" not in resp.headers.get("content-type", ""):
        print("[SKIP] Conținut non-HTML.")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "svg", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    if len(text.split()) < 30:
        print("[SKIP] Text insuficient.")
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
                    "text": chunk,
                }
            )
        )

    print(f"[INFO] {len(points)} bucăți → upload")
    batch_upload(points)


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


def safe_upsert(batch):
    for retry in range(5):
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
            return True
        except ResponseHandlingException:
            print(f"[WARN] Timeout Qdrant, retry {retry+1}/5…")
            time.sleep(1 + retry)
    return False


def batch_upload(points, batch_size=3):
    while points:
        batch = points[:batch_size]
        if safe_upsert(batch):
            print(f"[UPSERT] {len(batch)} puncte trimise.")
        points = points[batch_size:]


def main():
    print("=== START CRAWLER LIGHT ===")

    articles = fetch_latest_articles()
    print(f"[INFO] {len(articles)} articole detectate.")

    new_articles = [a for a in articles if not article_already_indexed(a)]
    print(f"[INFO] Articole noi: {len(new_articles)}")

    for url in new_articles:
        process_article(url)

    print("=== GATA LIGHT INDEX ===")


if __name__ == "__main__":
    main()

