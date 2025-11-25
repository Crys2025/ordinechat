"""
Crawler oficial pentru OrdineChat + Qdrant.
-------------------------------------------

Acest script trebuie rulat LOCAL, nu pe Render.

Face:
- crawling pe tot site-ul ordinesaudezordine.com
- curățare conținut (scripturi, share-links etc.)
- împărțire în bucăți (chunks)
- generare embeddings cu OpenAI
- upload automat în Qdrant

Comenzi:
    export OPENAI_API_KEY="..."
    export QDRANT_API_KEY="..."
    export QDRANT_URL="..."
    export COLLECTION_NAME="ordine_site"
    python crawler.py
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

# -------------------------
#   CONFIG
# -------------------------

BASE_URL = "https://ordinesaudezordine.com/"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# -------------------------
#  1. CREAȚI COLECȚIA
# -------------------------

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


# -------------------------
#  2. CRAWLING + CURĂȚARE
# -------------------------

BAD_LINK_PARTS = [
    "facebook.com", "twitter.com", "linkedin.com", "pinterest",
    "utm_", "share", "login", "wp-login", "password", "checkpoint",
    "mailto:", "tel:"
]

def get_links_and_text(url: str):
    print(f"[CRAWL] {url}")

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[E] Nu pot accesa {url}: {e}")
        return [], ("", "")  # evităm ValueError

    soup = BeautifulSoup(resp.text, "html.parser")

    # Eliminăm tag-urile inutile
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()

    # Textul principal al paginii
    text = soup.get_text(separator=" ", strip=True)

    # Dacă pagina e goală sau nerelevantă → o ignorăm
    if len(text.split()) < 30:
        return [], ("", "")

    # Extragem titlul
    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else url

    # Extragem linkuri interne curate
    clean_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        href = urljoin(url, href)  # absolutizare
        href, _ = urldefrag(href)  # scoatem #ancore

        # Ignorăm linkuri de share / login / externe
        if any(bad in href for bad in BAD_LINK_PARTS):
            continue

        if href.startswith(BASE_URL):
            clean_links.append(href)

    return list(set(clean_links)), (title, text)


# -------------------------
#  3. ÎMPĂRȚIRE TEXT
# -------------------------

def chunk_text(text: str, max_tokens: int = 400):
    words = text.split()
    chunks = []
    chunk = []

    for w in words:
        chunk.append(w)
        if len(chunk) >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


# -------------------------
# 4. EMBEDDINGS
# -------------------------

def embed_texts(texts):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]


# -------------------------
# 5. MAIN: ORCHESTRARE
# -------------------------

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

        # Extragem linkuri + conținut
        links, (title, text) = get_links_and_text(url)

        # Adăugăm linkuri noi la coadă
        for l in links:
            if l not in visited:
                to_visit.append(l)

        # Dacă textul e prea scurt sau gol → skip
        if not text or len(text.split()) < 30:
            continue

        # Împărțim în bucăți
        chunks = chunk_text(text, max_tokens=400)
        print(f"[INFO] {url} -> {len(chunks)} bucăți")

        # Embeddings
        vectors = embed_texts(chunks)

        # Creăm puncte Qdrant
        for vec, chunk in zip(vectors, chunks):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "url": url,
                    "title": title,
                    "text": chunk,
                },
            )
            buffer_points.append(point)

        # Trimitem progresiv
        if len(buffer_points) >= 40:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=buffer_points
            )
            print(f"[UPSERT] {len(buffer_points)} puncte trimise.")
            buffer_points = []

        time.sleep(0.3)  # evităm spam-ul

    # Trimitem ultimele puncte
    if buffer_points:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=buffer_points
        )
        print(f"[UPSERT FINAL] {len(buffer_points)} puncte trimise.")

    print("[GATA] Indexare completă în Qdrant.")


if __name__ == "__main__":
    main()

