from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifact_root: Path = Path(os.getenv("PHYSAI_ARTIFACT_ROOT", "artifacts"))
    database_url: str = os.getenv("PHYSAI_DATABASE_URL", "sqlite:///physai.db")
    faiss_index_dir: Path = Path(os.getenv("PHYSAI_FAISS_INDEX_DIR", "artifacts/faiss"))
    llm_base_url: str = os.getenv("PHYSAI_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
    llm_model: str = os.getenv("PHYSAI_LLM_MODEL", "meta/glm-5")
    llm_api_key_env: str = os.getenv("PHYSAI_LLM_API_KEY_ENV", "NVIDIA_API_KEY")
    random_seed: int = int(os.getenv("PHYSAI_RANDOM_SEED", "7"))


def get_settings() -> Settings:
    settings = Settings()
    settings.artifact_root.mkdir(parents=True, exist_ok=True)
    settings.faiss_index_dir.mkdir(parents=True, exist_ok=True)
    return settings
