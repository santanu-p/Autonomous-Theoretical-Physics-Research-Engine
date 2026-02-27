from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine
from sqlalchemy.engine import Engine

from physics_ai.config import get_settings
from physics_ai.utils import canonical_json


metadata = MetaData()

theories_table = Table(
    "theories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("name", String(255), nullable=False),
    Column("payload", JSON, nullable=False),
)

derivations_table = Table(
    "derivations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("stage", String(128), nullable=False),
    Column("canonical_hash", String(128), nullable=False),
    Column("payload", JSON, nullable=False),
)

background_runs_table = Table(
    "background_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
)

perturb_runs_table = Table(
    "perturb_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
)

qnm_runs_table = Table(
    "qnm_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("method", String(32), nullable=False),
    Column("omega_real", Float, nullable=False),
    Column("omega_imag", Float, nullable=False),
    Column("payload", JSON, nullable=False),
)

scan_jobs_table = Table(
    "scan_jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("name", String(255), nullable=False),
    Column("payload", JSON, nullable=False),
)

constraint_reports_table = Table(
    "constraint_reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("is_valid", Boolean, nullable=False),
    Column("payload", JSON, nullable=False),
)

novelty_reports_table = Table(
    "novelty_reports",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("novelty_score", Float, nullable=False),
    Column("payload", JSON, nullable=False),
)

paper_builds_table = Table(
    "paper_builds",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
)

events_table = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), nullable=False, index=True),
    Column("stage", String(64), nullable=False),
    Column("status", String(32), nullable=False),
    Column("message", Text, nullable=False),
    Column("payload", JSON, nullable=False),
)


def create_engine_from_settings() -> Engine:
    settings = get_settings()
    return create_engine(settings.database_url, future=True)


def init_db(engine: Engine | None = None) -> Engine:
    engine = engine or create_engine_from_settings()
    metadata.create_all(engine)
    return engine


def save_record(engine: Engine, table: Table, payload: dict[str, Any]) -> None:
    with engine.begin() as conn:
        conn.execute(table.insert().values(**payload))


def artifact_path(run_id: str, stage: str, filename: str) -> Path:
    settings = get_settings()
    path = settings.artifact_root / run_id / stage
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def write_artifact(run_id: str, stage: str, filename: str, content: str) -> Path:
    path = artifact_path(run_id, stage, filename)
    path.write_text(content, encoding="utf-8")
    return path


def write_json_artifact(run_id: str, stage: str, filename: str, payload: dict[str, Any]) -> Path:
    path = artifact_path(run_id, stage, filename)
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path
