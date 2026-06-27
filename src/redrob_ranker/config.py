"""
Central configuration: scoring weights, role taxonomy, and domain-term
vocabularies. Everything that encodes *judgment about the JD* lives here so the
scoring logic in features.py stays mechanical and the design is auditable in one
place (important for the Stage-5 defend-your-work interview).

The JD ("Senior AI Engineer — Founding Team @ Redrob AI") is decoded into:
  - which job TITLES signal a real fit vs a trap,
  - which DOMAIN TERMS in free-text career descriptions prove production
    retrieval/ranking/recsys experience,
  - which signals mark a candidate as available/hireable.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Structured-fit component weights (sum to 1.0). See features.structured_fit().
# ---------------------------------------------------------------------------
STRUCTURED_WEIGHTS = {
    "role": 0.32,        # title says AI/ML/retrieval/ranking  -> decisive vs keyword-stuffers
    "domain": 0.30,      # career DESCRIPTIONS prove retrieval/ranking/recsys work
    "product": 0.12,     # product company experience, not pure services
    "experience": 0.10,  # 5-9 year band (soft)
    "external": 0.06,    # github / open-source validation
    "location": 0.10,    # Pune/Noida/Tier-1 India or willing to relocate
}

# Blend of the rules channel and the retrieval (semantic) channel.
STRUCTURED_BLEND = 0.62
SEMANTIC_BLEND = 0.38

# Behavioral availability modifier maps to this multiplicative range.
BEHAVIOR_MIN = 0.55
BEHAVIOR_MAX = 1.12

# Honeypots are gated (multiplied) by this so they sink below the top-100.
HONEYPOT_GATE = 0.02

# Reciprocal-rank-fusion constant.
RRF_K = 60

# ---------------------------------------------------------------------------
# Role taxonomy. Matched as lowercased substrings against titles.
# Higher base score = closer to what the JD actually wants.
# ---------------------------------------------------------------------------
CORE_AI_TITLES = [
    "machine learning engineer", "ml engineer", "ai engineer",
    "applied scientist", "applied ml", "applied machine learning",
    "nlp engineer", "natural language", "research engineer",
    "recommendation systems engineer", "recommender", "recsys",
    "recommendation engineer", "search engineer", "search relevance",
    "relevance engineer", "ranking engineer", "personalization",
    "information retrieval", "deep learning engineer", "ml scientist",
    "machine learning scientist",
]
DATA_SCIENCE_TITLES = ["data scientist", "research scientist"]
ADJACENT_ML_TITLES = [
    "data engineer", "software engineer", "backend engineer", "back end",
    "platform engineer", "mlops", "ml platform", "software development engineer",
    "sde", "analytics engineer", "data platform",
]
OTHER_ENG_TITLES = [
    "frontend", "front end", "full stack", "fullstack", "mobile developer",
    "ios developer", "android developer", ".net developer", "java developer",
    "php developer", "devops", "cloud engineer", "network engineer",
    "qa engineer", "quality assurance", "test engineer", "web developer",
    "site reliability", "sre",
]
CV_SPEECH_ROBOTICS_TITLES = [
    "computer vision", "cv engineer", "vision engineer", "image",
    "speech", "robotics", "autonomous", "perception engineer",
]
# Explicitly non-engineering (JD's "do NOT want" + dataset keyword-stuffers).
OFF_ROLE_TITLES = [
    "marketing", "hr manager", "human resources", "recruiter", "talent",
    "accountant", "finance manager", "sales", "business development",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "graphic designer", "content writer", "copywriter", "operations manager",
    "project manager", "program manager", "business analyst",
    "customer support", "customer success", "product manager",
    "ui designer", "ux designer", "administrator", "consultant",
]

# Role category base scores.
ROLE_SCORES = {
    "core_ai": 1.00,
    "data_science": 0.85,
    "adjacent_ml": 0.45,
    "cv_speech_robotics": 0.30,
    "other_eng": 0.18,
    "off_role": 0.00,
    "unknown": 0.25,
}

# ---------------------------------------------------------------------------
# Domain-evidence vocabularies (matched in free-text career descriptions +
# summary). Grouped by weight: retrieval/ranking work is what the JD prizes.
# ---------------------------------------------------------------------------
RETRIEVAL_RANKING_TERMS = [
    "retrieval", "ranking", "rank model", "learning to rank", "learning-to-rank",
    "ltr", "recommendation", "recommender", "recsys", "semantic search",
    "vector search", "vector database", "embedding", "embeddings", "relevance",
    "ndcg", "mrr", "personalization", "nearest neighbor", "approximate nearest",
    " ann ", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "bm25", "two-tower", "two tower",
    "candidate generation", "re-ranking", "reranking", "search relevance",
]
NLP_LLM_TERMS = [
    "nlp", "natural language", "language model", " llm", "large language model",
    "transformer", "bert", "fine-tun", "rag", "retrieval-augmented",
    "sentence transformer", "sentence-transformers", "bge", " e5 ",
    "hugging face", "huggingface", "named entity", "text classification",
    "question answering", "summarization", "word2vec", "lora", "qlora", "peft",
]
PRODUCTION_SCALE_TERMS = [
    "production", "deployed", "real users", "at scale", "latency", "throughput",
    "serving", "inference", "real-time", "real time", "pipeline", "millions of",
    "high-traffic", "low-latency",
]
EVAL_TERMS = [
    "ndcg", "mrr", "mean average precision", "offline-online", "offline online",
    "offline evaluation", "a/b test", "ab test", "online experiment",
    "evaluation framework", "relevance labeling", "click-through", "ctr",
    "offline metrics",
]

DOMAIN_GROUP_WEIGHTS = {
    "retrieval_ranking": 3.0,
    "nlp_llm": 2.0,
    "production_scale": 1.5,
    "eval": 1.5,
}
# Saturation scale for domain score: domain = 1 - exp(-raw / SCALE).
DOMAIN_SATURATION = 6.0

# ---------------------------------------------------------------------------
# Indian IT-services firms. An entire career here is a JD disqualifier.
# ---------------------------------------------------------------------------
SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "hcl technologies", "mindtree",
    "ltimindtree", "lti", "l&t infotech", "mphasis", "hexaware", "birlasoft",
    "zensar", "mastek", "syntel", "igate", "persistent systems", "coforge",
    "ntt data", "dxc", "igate global",
}

# ---------------------------------------------------------------------------
# Location preferences (JD: Pune/Noida preferred; Tier-1 India welcome).
# ---------------------------------------------------------------------------
TOP_LOCATIONS = {"pune", "noida"}
TIER1_INDIA = {
    "hyderabad", "mumbai", "delhi", "new delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "chennai", "kolkata", "ahmedabad", "noida",
    "pune", "delhi ncr", "ncr",
}
INDIA_HINTS = {
    "india", "telangana", "maharashtra", "karnataka", "tamil nadu", "kerala",
    "gujarat", "haryana", "uttar pradesh", "west bengal", "odisha", "punjab",
    "rajasthan", "madhya pradesh", "andhra pradesh", "chandigarh",
    "trivandrum", "kochi", "coimbatore", "indore", "bhubaneswar", "vizag",
    "nagpur", "jaipur", "lucknow",
}

# ---------------------------------------------------------------------------
# JD "ideal profile" query used as the semantic-channel query vector.
# ---------------------------------------------------------------------------
JD_QUERY_TEXT = (
    "senior ai machine learning engineer building embeddings based retrieval "
    "ranking and recommendation systems in production at a product company. "
    "hybrid search dense and sparse retrieval vector database faiss pinecone "
    "elasticsearch. learning to rank relevance evaluation ndcg mrr map offline "
    "online a/b testing. nlp information retrieval llm fine-tuning rag deployed "
    "to real users at scale strong python evaluation frameworks for ranking."
)
