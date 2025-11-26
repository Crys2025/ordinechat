import os

# 🔥 Ștergem proxy-urile înainte să importăm OpenAI
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(key, None)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient

# 🔥 Importăm OpenAI AICI, DUPĂ ce am curățat proxy-urile
from openai import OpenAI

OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

@app.get("/")
def home():
    return {"status": "ok", "message": "OrdineBot backend online"}

@app.post("/ask")
def ask(question: Question):

    emb = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question.query
    )
    vector = emb.data[0].embedding

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=5
    )

    if not hits:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=f"Răspunde ca OrdineBot: {question.query}"
        )
        return {"answer": resp.output_text}

    context = ""
    for h in hits:
        p = h.payload or {}
        context += f"Titlu: {p.get('title')}\nURL: {p.get('url')}\nText: {p.get('text')}\n\n---\n\n"

    system = (
        "Tu ești OrdineBot, asistentul oficial al site-ului. "
        "Răspunzi cald, prietenos. "
        "Folosești DOAR informațiile din context. "
        "Dacă nu este clar, spui că nu apare în articole."
    )

    prompt = f"Context:\n{context}\nÎntrebare: {question.query}\nRăspuns:"

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
    )

    return {"answer": resp.output_text}





