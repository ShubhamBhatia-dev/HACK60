from llama_cpp import Llama
from config import phi_prompt

# ── Model registry ────────────────────────────────────────────────────────────
# Maps model key → path. Add more here without touching serve.py.
MODELS = {
    "qwen":      "./models/qwen_model.gguf",
    "phi":       "./models/phi_model.gguf",
    "tinyllama": "./models/tinyllama-jd.gguf",
}

DEFAULT_MODEL = "qwen"

# Lazy-loaded cache — each model is loaded only once on first use.
_cache: dict[str, Llama] = {}


def get_llm(model_key: str = DEFAULT_MODEL) -> Llama:
    key = model_key if model_key in MODELS else DEFAULT_MODEL
    if key not in _cache:
        _cache[key] = Llama(
            model_path=MODELS[key],
            n_gpu_layers=-1,
            verbose=False,
            n_threads=8,
            n_batch=512,
        )
    return _cache[key]


def stream_phi(data, context='', model_key: str = DEFAULT_MODEL):
    llm = get_llm(model_key)
    prompt = phi_prompt(data, context).strip() + "\n#"

    yield "#"

    for token in llm(
        prompt=prompt,
        stop=["<|end|>", "<|user|>", "<|system|>"],
        echo=False,
        max_tokens=800,
        temperature=0.1,
        stream=True,
    ):
        text = token["choices"][0]["text"]
        if text:
            yield text