import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from qdrant_client import QdrantClient

OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

app = FastAPI()

# Serve static files (ordinebot.js)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ordinesaudezordine.com",
        "https://www.ordinesaudezordine.com"
    ],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "ok", "message": "OrdineChat backend online."}


@app.post("/ask")
def ask(question: Question):

    # 1. Generate embedding for user question
    emb = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question.query
    )
    query_vector = emb.data[0].embedding

    # 2. Qdrant search with fallback
    try:
        search_result = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5,
            with_payload=True
        )
    except:
        search_result = qdrant.search_points(
            collection_name=COLLECTION_NAME,
            vector=query_vector,
            limit=5,
            with_payload=True
        )

    # 3. If nothing found in Qdrant → fallback GPT only
    if not search_result:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=f"Răspunde politicos la întrebarea: {question.query}"
        )
        return {"answer": resp.output_text}

    # 4. Build context from Qdrant results
    context_parts = []
    for point in search_result:
        payload = point.payload or {}
        piece = f"""
TITLU: {payload.get('title', '')}
URL: {payload.get('url', '')}
CONȚINUT: {payload.get('text', '')}
"""
        context_parts.append(piece.strip())

    context = "\n\n---\n\n".join(context_parts)

    # 5. System + User prompts
    system_prompt = (
        "Tu ești OrdineBot, asistentul oficial al site-ului ordinesaudezordine.com. "
        "Răspunzi cald, prietenos, profesionist, ca autoarea blogului. "
        "Folosești strict contextul oferit. "
        "Dacă contextul nu conține informația cerută, spune politicos că nu există în articole."
    )

    user_prompt = f"Context:\n{context}\n\nÎntrebare: {question.query}"

    # 6. Generate response using OpenAI Responses API
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {"answer": resp.output_text}
