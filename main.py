import os

# 🔥 Ștergem proxy-urile înainte să importăm OpenAI (ca să nu dea eroare pe server)
for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(key, None)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient

from openai import OpenAI

# 🔧 Config modele + colecție
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ordine_site")

# 🔑 Clienți OpenAI + Qdrant
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# 🚀 FastAPI app
app = FastAPI()

# CORS – permite apeluri din WordPress / alt domeniu
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servire fișiere statice (ordinebot.js etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")


# 📩 Schema request – TRIMITEM ÎNTREAGA CONVERSAȚIE
class Question(BaseModel):
    messages: list  # [{role: "user"/"assistant", content: "..."}, ...]


@app.get("/")
def home():
    return {"status": "ok", "message": "OrdineBot backend online"}


@app.post("/ask")
def ask(question: Question):
    """
    Endpoint-ul principal.
    Primește toată conversația (messages) și folosește:
    - ultimul mesaj de la user pentru căutarea în Qdrant
    - toată conversația ca memorie pentru model
    """

    # 🧠 Memorie conversațională – extragem ultimul mesaj de la user
    conversation_history = question.messages
    last_user_messages = [m for m in conversation_history if m.get("role") == "user"]

    if not last_user_messages:
        return {"answer": "Nu există un mesaj de utilizator în conversație."}

    current_query = last_user_messages[-1]["content"]

    # 📌 Embedding pe ÎNTREBAREA CURENTĂ
    emb = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=current_query,
    )
    vector = emb.data[0].embedding

    # 🔍 Căutare în Qdrant
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=5,
    )

    # ❗ Dacă nu găsim nimic în Qdrant → răspundem explicit
    if not hits:
        return {"answer": "Nu există informații despre asta pe site."}

    # 🧱 Construim contextul din articole
    context = ""
    for h in hits:
        payload = h.payload or {}
        context += (
            f"Titlu: {payload.get('title')}\n"
            f"URL: {payload.get('url')}\n"
            f"Text: {payload.get('text')}\n\n---\n\n"
        )

    # 🧠 Prompt de sistem – OrdineBot + memorie conversațională
    system = (
        "Ești OrdineBot, un asistent care răspunde STRICT pe baza articolelor "
        "de pe site-ul ordinesaudezordine.com/. "
        "Ai memorie conversațională: folosești întrebările și răspunsurile anterioare "
        "ca să deduci la ce se referă utilizatorul când spune, de exemplu, "
        "'dă-mi linkul' sau 'arată-mi articolul'. "
        "Nu inventezi informații. Nu adaugi opinii personale. "
        "Răspunzi foarte concis, 1-3 fraze maxim. "
        "DACĂ întrebarea nu are răspuns în context, spui exact: "
        "'Nu există informații despre asta pe site.' "
        "Nu folosești generalități, nu deviezi de la context."
    )

    # 🧠 Trimitem către model:
    # - instrucțiunile (system)
    # - contextul din articole (alt system)
    # - toată conversația user ↔ bot
    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"Context din articolele de pe site:\n{context}"},
    ] + conversation_history

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )

    return {"answer": resp.choices[0].message.content}
