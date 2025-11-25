from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from chromadb.config import Settings

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./db"))
collection = chroma.get_collection("site_content")

class Question(BaseModel):
    query: str

@app.post("/ask")
def ask(question: Question):
    results = collection.query(query_texts=[question.query], n_results=4)

    context = ""
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        context += f"<ARTICOL> {doc}\nLINK: {meta['url']}\n\n"

    prompt = f"""Tu ești OrdineBot, asistentul oficial al site‑ului ordinesaudezordine.com.
Răspunde cald, profesionist și oferă link-uri exacte la articole.
Folosește doar informațiile din context.

Context:
{context}

Întrebare: {question.query}
Răspuns:
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return {"answer": response.output_text}
