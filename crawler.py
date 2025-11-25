"""
Crawler oficial pentru OrdineChat + Qdrant.
Versiune FINALĂ – stabilă, rezistentă la toate erorile.

- ignoră linkurile de share
- ignoră redirecturi
- ignoră pagini ciudate (r.php, ?share=)
- nu crăpă pe 404/403
- returnează ÎNTOTDEAUNA valori valide
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

BASE_URL = "https://ordinesaudezordine.com/"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

BAD_LINK_PARTS = [
    "facebook.com", "twitter.com", "linkedin.com", "pinterest",
    "utm_", "share", "login", "wp-login", "password", "checkpoint",
    "mailto:", "tel:", "r.php", "redirect", "wp-json"
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


# -------------------------------------------------------
# FUNCTIE BLINDATĂ - NU VA RETURNA NICIODATĂ VALORI INVALIDATE
# -------------------------------------------------------
def get_links_and_text(url: str):
    print(f"[CRAWL] {url}")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[E] Nu pot accesa {url}: {e}")
        return [], ("", "")  # <-- forma corectă, nu crăpă

    soup = BeautifulSoup(resp.text, "html.parser")

    # Eliminăm taguri fără conținut util
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    if len(text.split()) < 30:
        return [], ("", "")

    # Titlu
    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else url

    # Linkuri curate
    clean_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        href = urljoin(url, href)
        href, _ = urldefrag(href)

        # Ignorăm linkurile nedorite
        if any(bad in href.lower() for bad in BAD_LINK_PARTS):
            continue

        if href.startswith(BASE_URL):
            clean_links.append(href)

    # RETURN GARANTAT VALID
    return list(set(clean_links)), (title, text)


def chunk_text(text: str, max_tokens: int = 400):
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
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


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

        # Adăugăm linkuri noi
        for l in links:
            if l not in visited:
                to_visit.append(l)

        # Dacă pagina nu are conținut, continuăm
        if not text or len(text.split()) < 30:
            continue

        # Spargem în bucăți
        chunks = chunk_text(text)
        print(f"[INFO] {url} -> {len(chunks)} bucăți")

        vectors = embed_texts(chunks)

        # Construim puncte Qdrant
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

        # Trimitem la Qdrant
        if len(buffer_points) >= 40:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=buffer_points
            )
            print(f"[UPSERT] {len(buffer_points)} puncte trimise.")
            buffer_points = []

        time.sleep(0.3)

    # Trimitem restul
    if buffer_points:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=buffer_points
        )
        print(f"[UPSERT FINAL] {len(buffer_points)} puncte trimise.")

    print("[GATA] Indexare completă în Qdrant.")


if __name__ == "__main__":
    main()


