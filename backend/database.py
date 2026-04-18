from pymongo import MongoClient
from config import dburl

client = MongoClient(dburl)
db = client["slm_app"]

users_col = db["users"]       # user accounts
sessions_col = db["sessions"] # token blacklisting (future)

# Each record = one job description session per user
# Stores the "best" tuple for fine-tuning: prompt → slm → user_edit → llm
jobs_col = db["jobs"]

# Indexes
users_col.create_index("email", unique=True)
jobs_col.create_index("job_id", unique=True)
jobs_col.create_index("user_id")          # query by user for history
