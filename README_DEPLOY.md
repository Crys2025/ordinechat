# OrdineChat + Qdrant backend

## Environment variables (Render)

Setează următoarele variabile în Render:

- `OPENAI_API_KEY`
- `QDRANT_API_KEY`
- `QDRANT_URL` = endpoint-ul Qdrant (ex: https://...cloud.qdrant.io:6333)
- `COLLECTION_NAME` = ordine_site

## Pornire locală

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Rulare crawler (LOCAL)

```bash
export OPENAI_API_KEY="..."
export QDRANT_API_KEY="..."
export QDRANT_URL="..."
export COLLECTION_NAME="ordine_site"
python crawler.py
```

Crawlerul va scana https://ordinesaudezordine.com/ și va popula colecția Qdrant.
