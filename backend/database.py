from pymongo import MongoClient, ASCENDING, DESCENDING
from config import dburl

client = MongoClient(dburl)
db = client["slm_app"]

users_col    = db["users"]         # user accounts
sessions_col = db["sessions"]      # token blacklisting (future)
jobs_col     = db["jobs"]          # one record per JD session
training_col = db["training_runs"] # one record per training run (metrics, config)
meta_col     = db["training_meta"] # singleton docs (e.g. latest_model path)

# ── Indexes ────────────────────────────────────────────────────────────────────
users_col.create_index("email", unique=True)
jobs_col.create_index("job_id", unique=True)
jobs_col.create_index("user_id")
jobs_col.create_index([("updated_at", DESCENDING)])
training_col.create_index([("started_at", DESCENDING)])
