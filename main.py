import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
from qdrant_client import QdrantClient

# -----------------------
# CONFIG
# -----------------------
OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

openai.api_key = os.getenv("OPENAI_API_KEY")

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

app = FastAPI()

# CORS pentru WordPress
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
    return {"status": "online", "message": "OrdineBot API merge!"}


@app.post("/ask")
def ask(question: Question):
    # Embedding pentru întrebare
    emb = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question.query
    )
    query_vector = emb.data[0].embedding

    # căutare Qdrant
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    # fallback dacă nu găsim articole
    if not results:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Ești OrdineBot, asistentul site-ului."},
                {"role": "user", "content": question.query},
            ]
        )
        return {"answer": resp.choices[0].message["content"]}

    # construim context
    context_texts = []
    for r in results:
        p = r.payload or {}
        context_texts.append(
            f"Titlu: {p.get('title','')}\nURL: {p.get('url','')}\nText: {p.get('text','')}"
        )

    context = "\n\n----\n\n".join(context_texts)

    system_prompt = (
        "Tu ești OrdineBot, asistentul oficial al site-ului ordinesaudezordine.com. "
        "Răspunzi cald, prietenos și explici clar. "
        "Folosește doar informațiile din context. "
        "Dacă nu este clar, spune că nu e menționat în articole."
    )

    full_prompt = f"Context:\n{context}\n\nÎntrebare: {question.query}\nRăspuns:"

    # generăm răspunsul final
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
    )

    return {"answer": resp.choices[0].message["content"]}


