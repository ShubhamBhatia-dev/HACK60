from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo.errors import DuplicateKeyError
from datetime import datetime
import asyncio
import json

from inference import stream_phi, DEFAULT_MODEL, MODELS
from database import users_col, jobs_col, training_col
from auth import hash_password, verify_password, create_token, get_current_user
from gemini import stream_gemini , build_gemini_prompt

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




# ── Streaming generation via SSE ─────────────────────────────────────────────

@app.get("/llm/stream")
async def llm_stream(
    query: str,
    useLLM: bool = False,
    slm_response: str = "",
    model: str = DEFAULT_MODEL,
    current_user: dict = Depends(get_current_user)
):
    if useLLM:
        
        enhance_prompt = build_gemini_prompt(query, slm_response)
        return StreamingResponse(
            gemini_sse_stream(enhance_prompt),
            media_type="text/event-stream",
            headers=sse_headers()
        )

    # Queue bridges the sync llama thread → async event loop
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def run_inference():
        """Runs in a thread; pushes tokens into the queue."""
        try:
            for token in stream_phi(query, context=slm_response, model_key=model):
                loop.call_soon_threadsafe(queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    async def token_stream():
        full_text = []
        loop.run_in_executor(None, run_inference)

        while True:
            token = await queue.get()
            if token is None:   
                break
            full_text.append(token)
            yield f"data: {json.dumps({'token': token, 'source': 'SLM'})}\n\n"

        yield f"data: {json.dumps({'done': True, 'full': ''.join(full_text), 'source': 'SLM'})}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers=sse_headers()
    )


async def gemini_sse_stream(query: str):
    """Gemini is already a sync generator — same queue pattern."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def run_gemini():
        try:
            for chunk in stream_gemini(query):
                loop.call_soon_threadsafe(queue.put_nowait, chunk)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def _stream():
        full_text = []
        loop.run_in_executor(None, run_gemini)
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            full_text.append(chunk)
            yield f"data: {json.dumps({'token': chunk, 'source': 'LLM'})}\n\n"
        yield f"data: {json.dumps({'done': True, 'full': ''.join(full_text), 'source': 'LLM'})}\n\n"

    async for item in _stream():
        yield item


def sse_headers():
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }








# ------------------------------------------------------------------------
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
    print(docs)
    return docs


# ── Root health check ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok"}


# ── Training runs (public — anyone can view metrics) ─────────────────────────
@app.get("/training-runs")
def get_training_runs(limit: int = 20):
    docs = list(
        training_col.find({}, {"_id": 0}).sort("started_at", -1).limit(limit)
    )
    return docs


# ── Available SLM models ──────────────────────────────────────────────────────
@app.get("/models")
def list_models():
    return {"models": list(MODELS.keys()), "default": DEFAULT_MODEL}





