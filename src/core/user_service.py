import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("SMART_BANKING_DB", "data/smart_banking.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'active'
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response_status TEXT,
                confidence REAL,
                latency REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
            """)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_user(
    username: str, email: str, password: str, user_id: Optional[str] = None
) -> str:
    ensure_schema()
    uid = (
        user_id
        or f"user_{hashlib.sha256(f'{username}:{email}'.encode('utf-8')).hexdigest()[:12]}"
    )
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO users (user_id, username, email, password_hash, created_at, status) VALUES (?, ?, ?, ?, ?, 'active')",
            (
                uid,
                username,
                email,
                hash_password(password),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return uid


def verify_user(username: str, password: str) -> Optional[str]:
    ensure_schema()
    with _connect() as conn:
        row = conn.execute(
            "SELECT user_id FROM users WHERE username = ? AND password_hash = ?",
            (username, hash_password(password)),
        ).fetchone()
    return row["user_id"] if row else None


def record_query(
    request_id: str,
    user_id: str,
    query: str,
    response_status: str,
    confidence: float,
    latency: float,
) -> None:
    ensure_schema()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO query_history (request_id, user_id, query, response_status, confidence, latency, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                user_id,
                query,
                response_status,
                confidence,
                latency,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def create_conversation(user_id: str, conversation_id: Optional[str] = None) -> str:
    ensure_schema()
    cid = (
        conversation_id
        or f"conv_{hashlib.sha256(f'{user_id}:{datetime.now(timezone.utc).isoformat()}'.encode('utf-8')).hexdigest()[:16]}"
    )
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (conversation_id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (
                cid,
                user_id,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return cid


def get_user(user_id: str) -> Optional[sqlite3.Row]:
    ensure_schema()
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
