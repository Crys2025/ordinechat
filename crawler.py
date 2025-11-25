"""
Script de crawling + indexare în Qdrant.

Rulează-l local (nu pe Render) când vrei să reindexezi site-ul:
    export OPENAI_API_KEY="..."
    export QDRANT_URL="..."
    export QDRANT_API_KEY="..."
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

BASE_URL = "https://ordinesaudezordine.com/"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

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
        return [], ""

    soup = BeautifulSoup(resp.text, "html.parser")

    # extragem textul principal
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # extragem link-urile interne
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        href = urljoin(url, href)
        href, _ = urldefrag(href)  # scoate #ancorele
        if href.startswith(BASE_URL):
            links.append(href)

    # titlu
    title_tag = soup.find("title")
    title = title_tag.text.strip() if title_tag else url

    return list(set(links)), (title, text)

def chunk_text(text: str, max_tokens: int = 800):
    # împărțim textul în bucăți aproximative
    words = text.split()
    chunk = []
    chunks = []
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
    vectors = [d.embedding for d in resp.data]
    return vectors

def main():
    create_collection_if_not_exists()

    visited = set()
    to_visit = [BASE_URL]
    all_points = []

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

        chunks = chunk_text(text, max_tokens=400)
        print(f"[INFO] {url} -> {len(chunks)} bucăți")

        vectors = embed_texts(chunks)

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
            all_points.append(point)

        # trimitem progresiv spre Qdrant
        if len(all_points) >= 50:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=all_points
            )
            print(f"[UPSERT] {len(all_points)} puncte trimise.")
            all_points = []

        time.sleep(1)  # să nu bombardăm serverul

    if all_points:
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=all_points
        )
        print(f"[UPSERT FINAL] {len(all_points)} puncte trimise.")

    print("[GATA] Indexare completă.")

if __name__ == "__main__":
    main()
