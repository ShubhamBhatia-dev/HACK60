from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo.errors import DuplicateKeyError
from datetime import datetime
import asyncio
import json

from inference import chatWithMe
from database import users_col, jobs_col
from auth import hash_password, verify_password, create_token, get_current_user

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic schemas ──────────────────────────────────────────────────────────
class SignupBody(BaseModel):
    name: str
    email: str
    password: str

class LoginBody(BaseModel):
    email: str
    password: str

class SaveJobBody(BaseModel):
    job_id: str
    prompt: str
    slm_output: str
    llm_output: str | None = None
    user_edited: str | None = None  # set only if user manually edited

# ── Auth routes ───────────────────────────────────────────────────────────────
@app.post("/signup")
def signup(body: SignupBody):
    try:
        result = users_col.insert_one({
            "name":     body.name,
            "email":    body.email,
            "password": hash_password(body.password),
        })
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(result.inserted_id)
    token   = create_token(user_id, body.email)
    return {"token": token, "user": {"id": user_id, "name": body.name, "email": body.email}}


@app.post("/login")
def login(body: LoginBody):
    doc = users_col.find_one({"email": body.email})
    if not doc or not verify_password(body.password, doc["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user_id = str(doc["_id"])
    token   = create_token(user_id, body.email)
    return {"token": token, "user": {"id": user_id, "name": doc["name"], "email": body.email}}


# ── Generation route ──────────────────────────────────────────────────────────
# Query params used so long/multiline prompts don't break the URL
@app.get("/llm")
def llm_op(query: str, useLLM: bool = False, current_user: dict = Depends(get_current_user)):
    if useLLM:
        # Placeholder — swap with real LLM call when ready
        op     = f"[LLM] Enhanced version of:\n\n{query}"
        source = "LLM"
    else:
        op     = chatWithMe(query)
        source = "SLM"
    return {"model_output": op, "source": source}

# ── Streaming generation via SSE ─────────────────────────────────────────────
@app.get("/llm/stream")
async def llm_stream(
    query: str,
    useLLM: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """
    Server-Sent Events streaming endpoint.
    Runs the model in a thread (non-blocking), then streams result word-by-word.
    Each message: data: {"token": "...", "source": "SLM"}
    Final message: data: {"done": true, "full": "...", "source": "SLM"}
    """
    if useLLM:
        full_text = f"[LLM] Enhanced version of:\n\n{query}"
        source    = "LLM"
    else:
        # Run blocking chatWithMe in thread pool — keeps event loop free
        full_text = await asyncio.to_thread(chatWithMe, query)
        source    = "SLM"

    async def token_stream():
        words = full_text.split(' ')
        for i, word in enumerate(words):
            sep   = ' ' if i < len(words) - 1 else ''
            chunk = json.dumps({"token": word + sep, "source": source})
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.02)   # 40 ms/word ≈ 25 words/sec
        # Done sentinel with full text for client verification
        yield f"data: {json.dumps({'done': True, 'full': full_text, 'source': source})}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        }
    )

@app.post("/save-job")
def save_job(body: SaveJobBody, current_user: dict = Depends(get_current_user)):
    """
    Upserts one record per job_id.
    Fields stored:
      user_id     — who owns this
      prompt      — what the user asked for
      slm_output  — what the local SLM generated
      llm_output  — what the external LLM enhanced (nullable)
      user_edited — what the user manually edited (nullable)
      updated_at  — last save time (ISO string)

    This tuple will be used to fine-tune the SLM automatically once
    we have enough samples (future script).
    """
    jobs_col.update_one(
        {"job_id": body.job_id},
        {"$set": {
            "user_id":     current_user["sub"],
            "job_id":      body.job_id,
            "prompt":      body.prompt,
            "slm_output":  body.slm_output,
            "llm_output":  body.llm_output,
            "user_edited": body.user_edited,
            "updated_at":  datetime.utcnow().isoformat(),
        }},
        upsert=True
    )
    return {"success": True}


# ── History (all jobs for logged-in user) ─────────────────────────────────────
@app.get("/history")
def get_history(current_user: dict = Depends(get_current_user)):
    docs = list(
        jobs_col.find(
            {"user_id": current_user["sub"]},
            {"_id": 0}               # exclude MongoDB _id
        ).sort("updated_at", -1).limit(50)
    )
    return docs


# ── Root health check ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}
