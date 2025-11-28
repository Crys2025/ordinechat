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

# ⭐ ADĂUGAT — doar acestea
import smtplib
from email.mime.text import MIMEText


# ⭐ ADĂUGAT — configurare email administrator
ADMIN_EMAIL = "ionutf993@gmail.com"

# autentificare Yahoo SMTP
SMTP_USER = "crys_20010@yahoo.com"
SMTP_PASS = "Ionut1989@"   # <-- pune aici parola reală

SMTP_SERVER = "android.smtp.mail.yahoo.com"
SMTP_PORT = 465  # Yahoo folosește SSL


def send_missing_email(query):
    """Trimite email când nu există informații în Qdrant."""

    body = (
        f"Un utilizator a căutat următorul subiect în GemeniBot:\n\n"
        f"🔎 Căutare: {query}\n\n"
        f"❗ Dar nu există informații pe site.\n"
        f"👉 Ar fi util să adaugi conținut pe acest subiect."
    )

    msg = MIMEText(body)
    msg["Subject"] = "⚠️ GemeniBot – Subiect căutat fără rezultate"
    msg["From"] = SMTP_USER
    msg["To"] = ADMIN_EMAIL

    try:
        # Yahoo cere SSL direct
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, [ADMIN_EMAIL], msg.as_string())
        server.quit()

        print("📩 Email trimis administratorului.")

    except Exception as e:
        print("❌ Eroare trimitere email:", e)



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
    messages: list  


@app.get("/")
def home():
    return {"status": "ok", "message": "OrdineBot backend online"}


@app.post("/ask")
def ask(question: Question):

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

    # ❗ Dacă nu găsim nimic în Qdrant → răspundem + trimitem email
    if not hits:
        send_missing_email(current_query)
        return {"answer": f"Nu există informații despre {current_query} pe site."}

    # 🧱 Construim contextul din articole
    context = ""
    for h in hits:
        payload = h.payload or {}
        context += (
            f"Titlu: {payload.get('title')}\n"
            f"URL: {payload.get('url')}\n"
            f"Text: {payload.get('text')}\n\n---\n\n"
        )

    system = (
        "Ești OrdineBot, un asistent care răspunde STRICT pe baza articolelor "
        "de pe site-ul ordinesaudezordine.com/. "
        "Ai memorie conversațională: folosești întrebările și răspunsurile anterioare "
        "ca să deduci la ce se referă utilizatorul când spune expresii precum "
        "'dă-mi linkul' sau 'arată-mi articolul'. "
        "Nu inventezi informații. "
        "Răspunzi concis (1–3 fraze). "
        "Dacă informația nu apare în context, spune EXACT: "
        "'Nu există informații despre asta pe site.' "
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": f"Context din articolele de pe site:\n{context}"},
    ] + conversation_history

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
    )

    return {"answer": resp.choices[0].message.content}


