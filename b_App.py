"""
PersonalGPT - A RAG chatbot over your own notes.

Architecture:
  [ you ] --> app (FastAPI, this file) --> chroma (vector DB, separate container)
                     |
                     +--> OpenAI API (embeddings + chat completion)
"""
import os
import sys
import time

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="PersonalGPT", description="RAG over the docker notes")

# --- Fail loudly and clearly if the API key is missing ---------------------
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not API_KEY or API_KEY.startswith("sk-paste"):
    sys.exit(
        "\n[PersonalGPT] OPENAI_API_KEY is missing.\n"
        "  Fix: cp .env.example .env, put your real key in it, then\n"
        "       docker compose up -d --force-recreate\n"
    )

llm = OpenAI(api_key=API_KEY)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name="text-embedding-3-small",
)

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))


def connect_to_chroma(retries: int = 30, delay: int = 2):
    """
    Chroma's container takes a few seconds to boot. `depends_on` only waits for
    the container to START, not to be READY - so we retry here instead of
    crashing on the first refused connection.
    """
    for attempt in range(1, retries + 1):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()
            print(f"[PersonalGPT] Connected to chroma at {CHROMA_HOST}:{CHROMA_PORT}")
            return client
        except Exception as e:
            print(
                f"[PersonalGPT] Waiting for chroma "
                f"({attempt}/{retries}): {type(e).__name__}",
                flush=True,
            )
            time.sleep(delay)
    sys.exit(f"\n[PersonalGPT] Could not reach chroma at {CHROMA_HOST}:{CHROMA_PORT}\n")


chroma = connect_to_chroma()
collection = chroma.get_or_create_collection(name="notes", embedding_function=openai_ef)


class Question(BaseModel):
    query: str


@app.get("/")
def health():
    return {
        "status": "PersonalGPT is live 📚",
        "docs_indexed": collection.count(),
        "chroma_host": CHROMA_HOST,
        "chroma_port": CHROMA_PORT,
    }


@app.post("/ask")
def ask(q: Question):
    if collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Run: docker compose exec app python ingest.py",
        )

    # 1. RETRIEVE - find the most relevant chunks from the vector DB
    hits = collection.query(query_texts=[q.query], n_results=3)
    documents = hits.get("documents") or [[]]
    context = "\n\n---\n\n".join(documents[0])

    # 2. AUGMENT - stuff that context into the prompt
    system_prompt = (
        "You are PersonalGPT, a helpful teaching assistant. "
        "Answer the user's question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"CONTEXT:\n{context}"
    )

    # 3. GENERATE - let the LLM write the final answer
    resp = llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q.query},
        ],
    )

    return {
        "question": q.query,
        "answer": resp.choices[0].message.content,
        "sources_used": len(documents[0]),
    }




#(infused_venv) PS E:\02_PersonalGPT> docker compose up -d 
#[+] up 12/12
# ✔ Image chromadb/chroma:0.6.3 Pulled                                                                                                                                                             73.9s
#1 [internal] load local bake definitions
#1 reading from stdin 729B done
#1 DONE 0.0s

#2 [internal] load build definition from Dockerfile
#2 transferring dockerfile:
#2 transferring dockerfile: 528B 0.4s done
#2 DONE 1.2s

#3 [internal] load metadata for docker.io/library/python:3.12-slim
#3 ...

#4 [auth] library/python:pull token for registry-1.docker.io
#4 DONE 0.0s

#3 [internal] load metadata for docker.io/library/python:3.12-slim
#3 DONE 2.4s

#5 [internal] load .dockerignore
#5 transferring context: 125B 0.0s done
#5 DONE 0.0s

#6 [internal] load build context
#6 ...

#7 [1/5] FROM docker.io/library/python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
#7 resolve docker.io/library/python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 0.1s done
#7 DONE 0.1s

#6 [internal] load build context
#6 transferring context: 7.59kB 0.4s done
#6 DONE 0.4s

#8 [3/5] COPY requirements.txt .
#8 CACHED

#9 [2/5] WORKDIR /app
#9 CACHED

#10 [4/5] RUN pip install --no-cache-dir -r requirements.txt
#10 CACHED

#11 [5/5] COPY . .
#11 DONE 1.8s

#12 exporting to image
#12 exporting layers
#12 exporting layers 0.3s done
#12 exporting manifest sha256:fe6fc7f2514f5d50baa24dbaf8707a0726f7cc57ff36ea21a1a176beca0109eb 0.1s done
#12 exporting config sha256:61b908b779cd1c58512893f9a6d8634473fff6a878f9615e4687863428d94da4 0.1s done
#12 exporting attestation manifest sha256:3b445710c2522a64245c029cbddf8fa6edca110b77953dd7e061eeaa981be3fb 0.1s done
#12 exporting manifest list sha256:eb262adc10d7814544fdf74da719638d1ec446623932ba4e8ed19e8535c79e18
#12 exporting manifest list sha256:eb262adc10d7814544fdf74da719638d1ec446623932ba4e8ed19e8535c79e18 0.1s done
#12 naming to docker.io/library/02_personalgpt-app:latest 0.0s done
#12 unpacking to docker.io/library/02_personalgpt-app:latest
#12 unpacking to docker.io/library/02_personalgpt-app:latest 9.5s done
#12 DONE 10.4s

# [+] up 16/16g provenance for metadata file
#  ✔ Image chromadb/chroma:0.6.3       Pulled                                                                                                           73.9s
#  ✔ Image 02_personalgpt-app          Built                                                                                                            20.2s
#  ✔ Volume 02_personalgpt_chroma_data Created                                                                                                           0.0s
#  ✔ Container prsonal-gpt-chroma      Started                                                                                                           3.3s
#  ✔ Container personal-gpt-app        Started                                                                                                           2.9s


# (infused_venv) PS E:\02_PersonalGPT> docker compose exec app python a_Ingest.py 
# Failed to send telemetry event ClientStartEvent: capture() takes 1 positional argument but 3 were given
# Ingested 15 chunks from 3 file(s) ✅
# Collection now holds 15 chunks total.

