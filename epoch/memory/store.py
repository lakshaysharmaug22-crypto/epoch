"""Experiment memory: runs, trials (with lineage + provenance), hypotheses, failures (vector-searchable),
incidents, deployments and an event log. SQLite by default; Postgres + pgvector in docker-compose."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.types import TypeDecorator

from epoch.config import settings
from epoch.memory.embed import DIM, cosine, embed


def now() -> datetime:
    return datetime.now(UTC)


class Embedding(TypeDecorator):
    """pgvector `vector(384)` on Postgres, JSON list elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector

                return dialect.type_descriptor(Vector(DIM))
            except ImportError:
                pass
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return list(map(float, value)) if dialect.name != "postgresql" else np.asarray(value, dtype=np.float32)

    def process_result_value(self, value, dialect):
        return None if value is None else np.asarray(value, dtype=np.float32)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    group_id: Mapped[str | None] = mapped_column(String(64), index=True)
    workload: Mapped[str] = mapped_column(String(64), index=True)
    strategy: Mapped[str] = mapped_column(String(32))
    seed: Mapped[int] = mapped_column(Integer)
    budget: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)


class Trial(Base):
    __tablename__ = "trials"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    number: Mapped[int] = mapped_column(Integer)
    genome_id: Mapped[str] = mapped_column(String(16), index=True)
    genome: Mapped[dict] = mapped_column(JSON)
    origin: Mapped[str] = mapped_column(String(24))
    generation: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    violations: Mapped[list] = mapped_column(JSON, default=list)
    feasible: Mapped[bool] = mapped_column(Boolean, default=False)
    pareto: Mapped[bool] = mapped_column(Boolean, default=False)
    fidelity: Mapped[float] = mapped_column(Float, default=1.0)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    hv_after: Mapped[float] = mapped_column(Float, default=0.0)
    parents: Mapped[list] = mapped_column(JSON, default=list)
    parents_inferred: Mapped[bool] = mapped_column(Boolean, default=False)
    hypothesis_id: Mapped[str | None] = mapped_column(String(32))
    extras: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    error_kind: Mapped[str | None] = mapped_column(String(24))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Hypothesis(Base):
    __tablename__ = "hypotheses"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    source: Mapped[str] = mapped_column(String(16))
    statement: Mapped[str] = mapped_column(Text)
    mechanism: Mapped[str] = mapped_column(Text, default="")
    target: Mapped[str] = mapped_column(String(32), default="")
    genes: Mapped[list] = mapped_column(JSON, default=list)
    expected: Mapped[dict] = mapped_column(JSON, default=dict)
    proposals: Mapped[list] = mapped_column(JSON, default=list)
    verdict: Mapped[str] = mapped_column(String(16), default="pending")
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)


class Failure(Base):
    __tablename__ = "failures"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    trial_id: Mapped[str | None] = mapped_column(String(80))
    workload: Mapped[str] = mapped_column(String(64), default="")
    kind: Mapped[str] = mapped_column(String(24))
    message: Mapped[str] = mapped_column(Text)
    genome: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding = mapped_column(Embedding)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workload: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    status: Mapped[str] = mapped_column(String(24), default="detected")
    title: Mapped[str] = mapped_column(Text, default="")
    severity: Mapped[str] = mapped_column(String(8), default="sev2")
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    embedding = mapped_column(Embedding)


class Deployment(Base):
    __tablename__ = "deployments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    workload: Mapped[str] = mapped_column(String(64))
    genome_id: Mapped[str] = mapped_column(String(16))
    genome: Mapped[dict] = mapped_column(JSON)
    reason: Mapped[str] = mapped_column(Text, default="")
    incident_id: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="active")


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


def _row(o: Any) -> dict[str, Any]:
    d = {c.name: getattr(o, c.name) for c in o.__table__.columns}
    for k, v in list(d.items()):
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, np.ndarray):
            d.pop(k)
    return d


class Store:
    def __init__(self, url: str | None = None):
        self.url = url or settings().resolved_db_url
        kw: dict[str, Any] = {"future": True, "pool_pre_ping": True}
        if self.url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = create_engine(self.url, **kw)
        if self.url.startswith("sqlite"):
            @event.listens_for(self.engine, "connect")
            def _wal(dbapi_conn, _):  # concurrent readers (API) + writer (engine)
                dbapi_conn.execute("PRAGMA journal_mode=WAL")
                dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        if self.engine.dialect.name == "postgresql":
            with self.engine.begin() as c:
                c.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        self._lock = threading.Lock()

    @contextmanager
    def session(self):
        with self._lock, self.Session() as s:
            yield s
            s.commit()

    # ---- runs --------------------------------------------------------------------------------
    def create_run(self, **kw: Any) -> None:
        with self.session() as s:
            s.merge(Run(**kw))

    def finish_run(self, run_id: str, summary: dict, status: str = "complete") -> None:
        with self.session() as s:
            r = s.get(Run, run_id)
            r.status, r.summary, r.finished_at = status, summary, now()

    def runs(self, workload: str | None = None, group_id: str | None = None) -> list[dict]:
        with self.session() as s:
            q = select(Run).order_by(Run.created_at)
            if workload:
                q = q.where(Run.workload == workload)
            if group_id:
                q = q.where(Run.group_id == group_id)
            return [_row(r) for r in s.scalars(q)]

    def run(self, run_id: str) -> dict | None:
        with self.session() as s:
            r = s.get(Run, run_id)
            return _row(r) if r else None

    # ---- trials ------------------------------------------------------------------------------
    def add_trial(self, **kw: Any) -> None:
        with self.session() as s:
            s.merge(Trial(**kw))

    def trials(self, run_id: str) -> list[dict]:
        with self.session() as s:
            return [_row(t) for t in s.scalars(select(Trial).where(Trial.run_id == run_id).order_by(Trial.number))]

    def set_pareto(self, run_id: str, trial_ids: set[str]) -> None:
        with self.session() as s:
            for t in s.scalars(select(Trial).where(Trial.run_id == run_id)):
                t.pareto = t.id in trial_ids

    # ---- hypotheses --------------------------------------------------------------------------
    def add_hypothesis(self, **kw: Any) -> None:
        with self.session() as s:
            s.merge(Hypothesis(**kw))

    def update_hypothesis(self, hid: str, **kw: Any) -> None:
        with self.session() as s:
            h = s.get(Hypothesis, hid)
            for k, v in kw.items():
                setattr(h, k, v)

    def hypotheses(self, run_id: str | None = None) -> list[dict]:
        with self.session() as s:
            q = select(Hypothesis).order_by(Hypothesis.created_at)
            if run_id:
                q = q.where(Hypothesis.run_id == run_id)
            return [_row(h) for h in s.scalars(q)]

    # ---- failures (vector memory) ------------------------------------------------------------
    def add_failure(self, *, run_id: str, trial_id: str | None, workload: str, kind: str, message: str,
                    genome: dict) -> None:
        vec = embed([f"{kind} {message[-800:]} " + " ".join(f"{k}={v}" for k, v in genome.items())])[0]
        with self.session() as s:
            s.add(Failure(run_id=run_id, trial_id=trial_id, workload=workload, kind=kind, message=message[-4000:],
                          genome=genome, embedding=vec))

    def similar_failures(self, query: str, k: int = 5, workload: str | None = None) -> list[dict]:
        q = embed([query])[0]
        with self.session() as s:
            if self.engine.dialect.name == "postgresql":
                stmt = select(Failure).order_by(Failure.embedding.cosine_distance(q)).limit(k)  # type: ignore[attr-defined]
                if workload:
                    stmt = stmt.where(Failure.workload == workload)
                rows = list(s.scalars(stmt))
                return [{**_row(r), "similarity": float(cosine(q[None], r.embedding[None])[0, 0])} for r in rows]
            stmt = select(Failure)
            if workload:
                stmt = stmt.where(Failure.workload == workload)
            rows = list(s.scalars(stmt))
        if not rows:
            return []
        sims = cosine(q[None], np.vstack([r.embedding for r in rows]))[0]
        order = np.argsort(-sims)[:k]
        return [{**_row(rows[i]), "similarity": float(sims[i])} for i in order]

    def failures(self, run_id: str | None = None) -> list[dict]:
        with self.session() as s:
            q = select(Failure).order_by(Failure.id)
            if run_id:
                q = q.where(Failure.run_id == run_id)
            return [_row(f) for f in s.scalars(q)]

    # ---- incidents / deployments / events ----------------------------------------------------
    def upsert_incident(self, iid: str, *, workload: str, status: str, title: str, severity: str, data: dict) -> None:
        vec = embed([title + " " + str(data.get("rca", {}).get("summary", ""))])[0]
        with self.session() as s:
            s.merge(Incident(id=iid, workload=workload, status=status, title=title, severity=severity, data=data,
                             embedding=vec))

    def incidents(self) -> list[dict]:
        with self.session() as s:
            return [_row(i) for i in s.scalars(select(Incident).order_by(Incident.created_at))]

    def incident(self, iid: str) -> dict | None:
        with self.session() as s:
            i = s.get(Incident, iid)
            return _row(i) if i else None

    def deploy(self, *, workload: str, genome_id: str, genome: dict, reason: str, incident_id: str | None = None) -> None:
        with self.session() as s:
            for d in s.scalars(select(Deployment).where(Deployment.workload == workload, Deployment.status == "active")):
                d.status = "superseded"
            s.add(Deployment(workload=workload, genome_id=genome_id, genome=genome, reason=reason, incident_id=incident_id))

    def deployments(self, workload: str | None = None) -> list[dict]:
        with self.session() as s:
            q = select(Deployment).order_by(Deployment.ts)
            if workload:
                q = q.where(Deployment.workload == workload)
            return [_row(d) for d in s.scalars(q)]

    def add_event(self, kind: str, payload: dict, run_id: str | None = None) -> None:
        with self.session() as s:
            s.add(Event(kind=kind, payload=payload, run_id=run_id))

    def events(self, since_id: int = 0, limit: int = 500) -> list[dict]:
        with self.session() as s:
            q = select(Event).where(Event.id > since_id).order_by(Event.id).limit(limit)
            return [_row(e) for e in s.scalars(q)]
