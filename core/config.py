"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Runtime settings, populated from environment variables."""

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    max_text_chars: int = int(os.getenv("MAX_TEXT_CHARS", "8000"))
    max_image_dim: int = int(os.getenv("MAX_IMAGE_DIM", "1024"))
    rag_dir: str = os.getenv("RAG_DIR", ".rag_store")
