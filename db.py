"""
db.py — SQLite helpers for Emotunes
Python 3.10 compatible
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Any, Iterable

DB_PATH = Path("user.db")

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash BLOB NOT NULL,
    age INTEGER,
    gender TEXT,
    languages TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    feedback TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS otp_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    otp TEXT NOT NULL,
    purpose TEXT NOT NULL,             -- signup, login_alt, reset
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH.as_posix(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)

# -------- Users --------
def create_user(email: str, password_hash: bytes) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email.lower().strip(), password_hash),
        )
        conn.commit()

def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),))
        return cur.fetchone()

def update_user_profile(email: str, *, age: Optional[int], gender: Optional[str], languages_csv: Optional[str]) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE users SET age=?, gender=?, languages=? WHERE email=?",
            (age, gender, languages_csv, email.lower().strip()),
        )
        conn.commit()

def update_user_password(email: str, password_hash: bytes) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE email=?",
            (password_hash, email.lower().strip()),
        )
        conn.commit()

# -------- Feedback --------
def add_feedback(user_id: int, feedback: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO feedback (user_id, feedback) VALUES (?, ?)",
            (user_id, feedback),
        )
        conn.commit()

def get_feedback(user_id: int) -> Iterable[sqlite3.Row]:
    with connect() as conn:
        cur = conn.execute("SELECT * FROM feedback WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
        yield from cur.fetchall()

# -------- OTP log (optional audit) --------
def log_otp(email: str, otp: str, purpose: str) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO otp_log (email, otp, purpose) VALUES (?,?,?)",
            (email.lower().strip(), otp, purpose),
        )
        conn.commit()
