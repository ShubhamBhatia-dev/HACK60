# latestPath.py
# Single source of truth for the currently active fine-tuned GGUF model.
# Written by the training pipeline after a successful export.
# Read by inference.py to pick the default model.

from pathlib import Path
from pymongo import MongoClient
from config import dburl

_client = MongoClient(dburl)
_meta_col = _client["slm_app"]["training_meta"]

DEFAULT_FALLBACK = "./models/qwen_model.gguf"


def get_latest_gguf() -> str:
    """Return the path of the latest fine-tuned GGUF, or the factory default."""
    doc = _meta_col.find_one({"_id": "latest_model"})
    if doc and doc.get("gguf_path"):
        p = Path(doc["gguf_path"])
        if p.exists():
            return str(p)
    return DEFAULT_FALLBACK


def set_latest_gguf(path: str) -> None:
    """Persist the path of a newly exported GGUF to MongoDB."""
    _meta_col.update_one(
        {"_id": "latest_model"},
        {"$set": {"_id": "latest_model", "gguf_path": path}},
        upsert=True,
    )
