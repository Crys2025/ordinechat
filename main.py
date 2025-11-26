import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest, Filter, FieldCondition, MatchValue, PointStruct

OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

# 🔥 IMPORTANT – forțăm clientul OpenAI vechi
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

# Qdrant
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

    # 1️⃣ Obținem embedding
    emb = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question.query
    )
    vector = emb.data[0].embedding

    # 2️⃣ Căutare în Qdrant (versiune corectă)
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5
    ).points

    # 3️⃣ Nu există rezultate → fallback LLM
    if not hits:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=f"Răspunde ca OrdineBot: {question.query}"
        )
        return {"answer": resp.output_text}

    # 4️⃣ Construim contextul
    context = ""
    for h in hits:
        p = h.payload or {}
        context += (
            f"Titlu: {p.get('title')}\n"
            f"URL: {p.get('url')}\n"
            f"Text: {p.get('text')}\n\n---\n\n"
        )

    # 5️⃣ Prompt final
    system = (
        "Tu ești OrdineBot, asistentul oficial al site-ului. "
        "Răspunzi cald, prietenos, scurt și clar. "
        "Folosești DOAR informațiile din context. "
        "Dacă ceva nu există în context, spui clar acest lucru."
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




