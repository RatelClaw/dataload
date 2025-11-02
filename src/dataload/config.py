import logging
from logging import Formatter
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Note: New configuration system available in dataload.config.embedding_config

# --- Environment Variables ---
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SECRET_NAME = os.environ.get("SECRET_NAME", "postgres/db-credentials")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
CONTENT_TYPE = os.environ.get("CONTENT_TYPE", "application/json")
DEFAULT_DIMENSION = int(os.environ.get("DEFAULT_DIMENSION", 1024))
# DEFAULT_DIMENSION=384 # for sentence-transformers/all-MiniLM-L6-v2
# DEFAULT_DIMENSION=1024 # for Amazon Titan bedrock
# DEFAULT_DIMENSION =768 # 3072  # for Gemini


DEFAULT_VECTOR_VALUE = float(os.environ.get("DEFAULT_VECTOR_VALUE", 0.0))

LOCAL_POSTGRES_HOST = os.environ.get("LOCAL_POSTGRES_HOST", "localhost")
LOCAL_POSTGRES_PORT = int(os.environ.get("LOCAL_POSTGRES_PORT", 5432))
LOCAL_POSTGRES_DB = os.environ.get("LOCAL_POSTGRES_DB", "vector_db")
LOCAL_POSTGRES_USER = os.environ.get("LOCAL_POSTGRES_USER", "postgres")
LOCAL_POSTGRES_PASSWORD = os.environ.get("LOCAL_POSTGRES_PASSWORD", "password")

# --- Logging Setup ---
logger = logging.getLogger("vector-dataloader")
logger.setLevel(logging.INFO)

# Clear default handlers to prevent duplicate logs in some environments
if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler()
handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


class JSONFormatter(Formatter):
    """A formatter for JSON-structured logs."""

    def format(self, record):
        return json.dumps(
            {
                "time": self.formatTime(record),
                "name": record.name,
                "level": record.levelname,
                "msg": record.getMessage(),  # Use getMessage() for safer access
            }
        )
