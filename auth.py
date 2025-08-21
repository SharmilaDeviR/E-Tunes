"""
auth.py — Signup/Login/OTP handling
Uses bcrypt for password hashing and simple SMTP for email OTP.
"""

from __future__ import annotations
import os, time, random, smtplib, ssl
from email.mime.text import MIMEText
from typing import Optional, Dict
import bcrypt
from dotenv import load_dotenv

from db import (
    init_db,
    create_user,
    get_user_by_email,
    update_user_password,
    log_otp,
)

# Load environment variables from .env
load_dotenv()

# In-memory OTP store: { email: {"otp": "123456", "exp": 1690000000.0, "purpose": "signup"} }
_OTP_CACHE: Dict[str, Dict[str, object]] = {}

OTP_TTL_SEC = 300  # 5 minutes

# ---------- Password hashing ----------
def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12))

def verify_password(plain: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed)
    except Exception:
        return False

# ---------- OTP ----------
def _generate_otp() -> str:
    return f"{random.randint(100000, 999999)}"

def send_otp(email: str, *, purpose: str = "signup") -> str:
    """
    Sends OTP via SMTP. Falls back gracefully if SMTP fails.
    Env vars required in .env:
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM (optional), SMTP_TLS ("true"/"false")
    """
    otp = _generate_otp()
    _OTP_CACHE[email] = {"otp": otp, "exp": time.time() + OTP_TTL_SEC, "purpose": purpose}
    try:
        _smtp_send(email, otp, purpose)
        print(f"✅ OTP sent to {email}")
    except Exception as e:
        print(f"⚠️ OTP send failed: {e}")
        # SMTP failed → caller (app.py) can decide whether to show Demo OTP
        raise
    finally:
        try:
            log_otp(email, otp, purpose)
        except Exception:
            pass
    return otp

def verify_otp(email: str, entered: str, *, purpose: Optional[str] = None) -> bool:
    rec = _OTP_CACHE.get(email)
    if not rec:
        return False
    if time.time() > float(rec["exp"]):
        _OTP_CACHE.pop(email, None)
        return False
    if purpose and rec.get("purpose") != purpose:
        return False
    ok = str(rec["otp"]) == str(entered).strip()
    if ok:
        _OTP_CACHE.pop(email, None)
    return ok

def _smtp_send(to_email: str, otp: str, purpose: str) -> None:
    """Send OTP via SMTP using .env configuration"""
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", user or "no-reply@emotunes")
    use_tls = os.getenv("SMTP_TLS", "true").lower() in ("1","true","yes")

    if not user or not pwd:
        raise ValueError("❌ SMTP credentials are missing in .env file")

    subject = f"Emotunes OTP ({purpose})"
    body = f"Your OTP is: {otp}\n\nIt will expire in 5 minutes."

    msg = MIMEText(body)
    msg["From"] = from_addr
    msg["To"] = to_email
    msg["Subject"] = subject

    with smtplib.SMTP(host, port) as server:
        if use_tls:
            server.starttls(context=ssl.create_default_context())
        server.login(user, pwd)
        server.sendmail(from_addr, [to_email], msg.as_string())

# ---------- High-level flows ----------
def signup(email: str, password: str) -> None:
    """
    Create user with hashed password. Caller should verify OTP separately.
    Raises if email already exists.
    """
    init_db()
    if get_user_by_email(email):
        raise ValueError("Email already exists")
    create_user(email, hash_password(password))

def login(email: str, password: str) -> bool:
    """Return True if credentials ok."""
    init_db()
    row = get_user_by_email(email)
    if not row:
        return False
    return verify_password(password, row["password_hash"])

def start_password_reset(email: str) -> str:
    """Send OTP to reset password. Returns OTP (for debug/UI fallback)."""
    if not get_user_by_email(email):
        raise ValueError("Email not registered")
    return send_otp(email, purpose="reset")

def complete_password_reset(email: str, otp: str, new_password: str) -> bool:
    """Verify OTP and set new password."""
    if not verify_otp(email, otp, purpose="reset"):
        return False
    update_user_password(email, hash_password(new_password))
    return True
