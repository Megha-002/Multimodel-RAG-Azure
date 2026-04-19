import os
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

# --- Groq ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))

# --- Hugging Face ---
HF_API_KEY = os.getenv("HF_API_KEY")
HF_EMBED_URL = os.getenv("HF_EMBED_URL")

# --- Pinecone ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "ragindex")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# --- MLflow / DagsHub ---
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# --- Grafana ---
GRAFANA_REMOTE_WRITE_URL = os.getenv("GRAFANA_REMOTE_WRITE_URL")
GRAFANA_PROM_USERNAME = os.getenv("GRAFANA_PROM_USERNAME")
GRAFANA_PROM_API_KEY = os.getenv("GRAFANA_PROM_API_KEY")

# --- RAG Config ---
TOP_K = int(os.getenv("TOP_K", 5))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))