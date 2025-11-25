"""
Crawler stabil pentru OrdineChat + Qdrant.
Optimizat 100% pentru Qdrant Cloud Free (batchuri mici + retry).
"""

import os
import time
import uuid
import requests
from urllib.parse import urljoin, urldefrag
from bs4 import BeautifulSoup
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException

BASE_URL = "https://ordinesaudezordine.com/"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=30.0  # timeout mare, dar sigur
)

# Linkuri care nu trebuie vizitate niciodată
BAD_LINK_PARTS = [
    "facebook.com", "twitter.com", "linkedin.com", "pinterest",
    "utm_", "share", "login", "wp-login", "password", "checkpoint",
    "r.php", "redirect", "wp-json", "mailto:", "tel:"
]

def create_collection_if_not_exists(dim: int = 1536):
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[OK] Colecția '{COLLECTION_NAME}' a fost creată.")
    else:
        print(f"[OK] Colecția '{COLLECTION_NAME}' există deja.")


def get_links_and_text(url: str):
    print(f"[CRAWL] {url}")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[E] Nu pot accesa {url}: {e}")
        return [], ("", "")

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    if len(text.split()) < 30:
        return [], ("", "")

    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else url

    clean_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        href = urljoin(url, href)
        href, _ = urldefrag(href)

        if any(bad in href.lower() for bad in BAD_LINK_PARTS):
            continue

        if href.startswith(BASE_URL):
            clean_links.append(href)

    return list(set(clean_links)), (title, text)


def chunk_text(text: str, max_tokens: int = 350):
    words = text.split()
    chunk, chunks = [], []

    for w in words:
        chunk.append(w)
        if len(chunk) >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def embed_texts(texts):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


# --------------------------
# QDRANT SAFE UPSERT (batch mic + retry)
# --------------------------
def safe_qdrant_upsert(points_batch):
    for retry in range(5):  # până la 5 încercări
        try:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )
            return True
        except ResponseHandlingException as e:
            print(f"[WARN] Qdrant timeout, retry {retry+1}/5...")
            time.sleep(1 + retry * 1.5)
    print("[FATAL] Qdrant nu a răspuns după 5 încercări.")
    return False


def main():
    create_collection_if_not_exists()

    visited = set()
    to_visit = [BASE_URL]
    buffer_points = []

    while to_visit:
        url = to_visit.pop()

        if url in visited:
            continue
        visited.add(url)

        links, (title, text) = get_links_and_text(url)

        for l in links:
            if l not in visited:
                to_visit.append(l)

        if not text or len(text.split()) < 30:
            continue

        chunks = chunk_text(text)
        print(f"[INFO] {url} -> {len(chunks)} bucăți")

        vectors = embed_texts(chunks)

        for vec, chunk in zip(vectors, chunks):
            buffer_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "url": url,
                        "title": title,
                        "text": chunk,
                    },
                )
            )

        # Trimitem în loturi de câte 3 puncte
        while len(buffer_points) >= 3:
            batch = buffer_points[:3]
            if safe_qdrant_upsert(batch):
                print(f"[UPSERT] 3 puncte trimise.")
            buffer_points = buffer_points[3:]

        time.sleep(0.2)

    # Trimitem restul
    while buffer_points:
        batch = buffer_points[:3]
        if safe_qdrant_upsert(batch):
            print(f"[UPSERT FINAL] {len(batch)} puncte trimise.")
        buffer_points = buffer_points[3:]

    print("[GATA] Indexare completă în Qdrant.")


if __name__ == "__main__":
    main()



