"""
CRAWLER FULL (INTELIGENT)
- Indexează TOT site-ul
- Sare peste paginile deja indexate (NO DUPLICATES)
- Acceptă DOAR HTML
- Ignoră fișiere media (mp4, jpg, pdf etc.)
- Perfect stabil pentru reindexări totale
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
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
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
    "facebook.com", "twitter.com", "linkedin.com", "pinterest", "utm_",
    "share", "login", "wp-login", "password", "checkpoint", "redirect",
    "r.php", "wp-json", "mailto:", "tel:"
]

MEDIA_EXT = [
    ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".mp4", ".mov", ".avi", ".mp3", ".webm",
    ".pdf", ".zip", ".rar", ".7z",
    ".doc", ".docx", ".xlsx", ".pptx"
]


def create_collection_if_not_exists(dim=1536):
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        print(f"[OK] Colecția '{COLLECTION_NAME}' a fost creată.")
    else:
        print(f"[OK] Colecția '{COLLECTION_NAME}' există deja.")


def is_url_indexed(url: str) -> bool:
    """Verifică dacă URL-ul are deja puncte în Qdrant."""
    scroll = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="url", match=MatchValue(value=url))]
        ),
        limit=1
    )
    return len(scroll[0]) > 0


def get_links_and_text(url: str):
    print(f"[CRAWL] {url}")

    # Ignorăm fișiere media
    if any(url.lower().endswith(ext) for ext in MEDIA_EXT):
        print("[SKIP] Fișier media.")
        return [], ("", "")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[E] Eroare accesare {url}: {e}")
        return [], ("", "")

    # Acceptăm DOAR HTML
    content_type = resp.headers.get("content-type", "")
    if "text/html" not in content_type:
        print(f"[SKIP] Non-HTML ({content_type})")
        return [], ("", "")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    if len(text.split()) < 30:
        return [], ("", "")

    title = soup.find("title").text.strip() if soup.find("title") else url

    # linkuri curate
    clean_links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(url, a["href"]).split("#")[0]

        if any(bad in href.lower() for bad in BAD_LINK_PARTS):
            continue

        if any(href.lower().endswith(ext) for ext in MEDIA_EXT):
            continue

        if href.startswith(BASE_URL):
            clean_links.append(href)

    return list(set(clean_links)), (title, text)


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


def safe_upsert(points_batch):
    for retry in range(5):
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            return True
        except ResponseHandlingException:
            print(f"[WARN] Timeout Qdrant, retry {retry+1}/5…")
            time.sleep(1 + retry)
    return False


def main():
    create_collection_if_not_exists()

    visited = set()
    to_visit = [BASE_URL]
    buffer = []

    while to_visit:
        url = to_visit.pop()

        if url in visited:
            continue
        visited.add(url)

        # ⭐ PAS NOU: dacă URL-ul există deja în Qdrant → îl sărim
        if is_url_indexed(url):
            print(f"[SKIP] Deja indexat: {url}")
            continue

        links, (title, text) = get_links_and_text(url)

        # adăugăm linkuri noi
        for l in links:
            if l not in visited:
                to_visit.append(l)

        if not text:
            continue

        chunks = chunk_text(text)
        vectors = embed_texts(chunks)

        for vec, chunk in zip(vectors, chunks):
            buffer.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={"url": url, "title": title, "text": chunk}
                )
            )

        # trimitem în loturi mici
        while len(buffer) >= 3:
            batch = buffer[:3]
            if safe_upsert(batch):
                print(f"[UPSERT] 3 puncte trimise.")
            buffer = buffer[3:]

    # restul
    while buffer:
        batch = buffer[:3]
        if safe_upsert(batch):
            print(f"[UPSERT FINAL] {len(batch)} puncte trimise.")
        buffer = buffer[3:]

    print("[GATA] Indexare FULL cu skip duplicări.")


if __name__ == "__main__":
    main()



