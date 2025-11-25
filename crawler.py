# Script simplu pentru scanarea site-ului și creare index
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.config import Settings

BASE_URL = "https://ordinesaudezordine.com/"

def crawl(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(BASE_URL):
            links.append(href)

    text = soup.get_text(separator=" ", strip=True)
    return text, links

def crawl_site(start_url):
    visited = set()
    to_visit = [start_url]
    pages = []

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue

        visited.add(url)
        try:
            text, links = crawl(url)
            pages.append((url, text))
            for link in links:
                if link not in visited:
                    to_visit.append(link)
        except:
            pass

    return pages

print("Crawling...")
pages = crawl_site(BASE_URL)
print("Pagini găsite:", len(pages))

chroma = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./db"))
try:
    chroma.delete_collection("site_content")
except:
    pass

collection = chroma.create_collection(name="site_content")

for idx, (url, text) in enumerate(pages):
    collection.add(ids=[str(idx)], documents=[text[:5000]], metadatas=[{"url": url}])

print("Indexare completă!")
