import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

OPENAI_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

app = FastAPI()

# CORS - permite apeluri de la site-ul tău
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # poți restricționa pe domeniul tău
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return {"status": "ok", "message": "OrdineChat Qdrant backend online."}

@app.post("/ask")
def ask(question: Question):
    # obținem embedding pentru întrebare
    emb_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question.query
    )
    query_vector = emb_response.data[0].embedding

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    if not search_result:
        # fallback: doar GPT, fără context (mai slab, dar nu moare)
        prompt = f"Tu ești OrdineBot, asistentul site-ului ordinesaudezordine.com. Răspunde politicos la întrebarea: {question.query}"
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt
        )
        return {"answer": resp.output_text}

    context_parts = []
    for point in search_result:
        payload = point.payload or {}
        text = payload.get("text", "")
        url = payload.get("url", "")
        title = payload.get("title", "")
        piece = f"TITLU: {title}\nURL: {url}\nCONȚINUT: {text}"
        context_parts.append(piece)

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "Tu ești OrdineBot, asistentul oficial al site-ului ordinesaudezordine.com. "
        "Răspunzi cald, prietenos, profesionist, ca autoarea blogului. "
        "Folosești DOAR informațiile din contextul dat. "
        "Dacă nu găsești ceva în context, spune politicos că nu e clar din articole. "
        "La final, când este relevant, sugerează 1-3 articole cu titlu și URL."
    )

    full_prompt = f"Context articole:\n{context}\n\nÎntrebare utilizator: {question.query}\nRăspuns OrdineBot:"

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]
    )

    return {"answer": response.output_text}
