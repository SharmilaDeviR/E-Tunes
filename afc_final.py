# app.py — Emotunes: Hear how you feel
# Python 3.10 / TensorFlow 2.13 compatible

import os
import base64
import time
import random
import smtplib
import ssl
import sqlite3
import datetime as dt
from email.mime.text import MIMEText
from typing import Dict, Optional, List

import bcrypt
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from dotenv import load_dotenv

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ===================== CONFIG =====================
APP_TITLE = "Emotunes — Hear how you feel"
DB_PATH = "user.db"
OTP_TTL_SEC = 300  # 5 minutes

def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

EMOTION_MODEL_PATH = _first_existing([os.path.join("models", "emotion_model.h5"), "emotion_model.h5"])
WEATHER_MODEL_PATH = _first_existing([os.path.join("models", "weather_model.h5"), "weather_model.h5"])

load_dotenv()

# ===================== STREAMLIT SETUP =====================
st.set_page_config(page_title="Emotunes", layout="centered", page_icon="🎧")

def set_background(img_path: str = "themebg.png"):
    if not os.path.exists(img_path):
        return
    with open(img_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    css = f"""
    <style>
    .stApp {{
        background: url('data:image/png;base64,{b64}') no-repeat center center fixed;
        background-size: cover;
    }}
    .block-container {{
        background: rgba(255,255,255,0.70);
        backdrop-filter: blur(6px);
        border-radius: 24px;
        padding: 2rem 2rem 3rem 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("themebg.png")

# ===================== DB LAYER =====================
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
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db_conn() as c:
        c.executescript(SCHEMA)

def create_user(email: str, password_hash: bytes):
    with db_conn() as c:
        c.execute("INSERT INTO users (email, password_hash) VALUES (?,?)", (email.lower().strip(), password_hash))
        c.commit()

def get_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with db_conn() as c:
        cur = c.execute("SELECT * FROM users WHERE email=?", (email.lower().strip(),))
        return cur.fetchone()

def update_user_profile(email: str, age: int, gender: str, languages_csv: str):
    with db_conn() as c:
        c.execute("UPDATE users SET age=?, gender=?, languages=? WHERE email=?",
                  (age, gender, languages_csv, email.lower().strip()))
        c.commit()

def update_user_password(email: str, password_hash: bytes):
    with db_conn() as c:
        c.execute("UPDATE users SET password_hash=? WHERE email=?", (password_hash, email.lower().strip()))
        c.commit()

def add_feedback(user_id: int, feedback: str):
    if not feedback.strip():
        return
    with db_conn() as c:
        c.execute("INSERT INTO feedback (user_id, feedback) VALUES (?,?)", (user_id, feedback.strip()))
        c.commit()

# ===================== AUTH / OTP =====================
def hash_password(plain: str) -> bytes:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12))

def verify_password(plain: str, hashed: bytes) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed)
    except Exception:
        return False

# Use session_state for OTP cache so it survives reruns
if "otp_cache" not in st.session_state:
    st.session_state.otp_cache = {}

def _generate_otp() -> str:
    return f"{random.randint(100000, 999999)}"

def _smtp_send(to_email: str, otp: str, purpose: str):
    host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM", user or "no-reply@emotunes")
    use_tls = os.getenv("SMTP_TLS", "true").lower() in ("1","true","yes")

    if not user or not pwd:
        raise RuntimeError("SMTP not configured")

    subject = f"Emotunes OTP ({purpose})"
    body = f"Your Emotunes OTP is: {otp}\n\nThis code expires in 5 minutes."

    msg = MIMEText(body)
    msg["From"] = from_addr
    msg["To"] = to_email
    msg["Subject"] = subject

    with smtplib.SMTP(host, port) as server:
        if use_tls:
            server.starttls(context=ssl.create_default_context())
        server.login(user, pwd)
        server.sendmail(from_addr, [to_email], msg.as_string())

def send_otp(email: str, purpose: str = "signup"):
    norm_email = email.strip().lower()
    otp = _generate_otp()
    st.session_state.otp_cache[norm_email] = {
        "otp": otp,
        "exp": time.time() + OTP_TTL_SEC,
        "purpose": purpose
    }
    try:
        _smtp_send(norm_email, otp, purpose)
        st.success(f"✅ OTP sent to {norm_email}. Check your inbox (valid 5 minutes).")
    except Exception as e:
        st.warning(f"⚠️ Could not send email: {e}")
        st.info(f"💡 Demo OTP for {norm_email}: **{otp}** (valid 5 min)")
    return otp

def verify_otp(email: str, entered: str, purpose: Optional[str] = None) -> bool:
    norm_email = (email or "").strip().lower()
    rec = st.session_state.otp_cache.get(norm_email)
    if not rec:
        return False
    if time.time() > float(rec["exp"]):
        st.session_state.otp_cache.pop(norm_email, None)
        return False
    if purpose and rec.get("purpose") != purpose:
        return False
    ok = str(rec["otp"]) == str(entered).strip()
    if ok:
        st.session_state.otp_cache.pop(norm_email, None)
    return ok

# ===================== MODELS =====================
EMOTION_LABELS = ['angry','disgust','fear','happy','neutral','sad','surprise']

def load_keras_model(path: Optional[str]):
    try:
        if path and os.path.exists(path):
            return tf.keras.models.load_model(path)
    except Exception as e:
        st.warning(f"Could not load model at {path}: {e}")
    return None

emotion_model = load_keras_model(EMOTION_MODEL_PATH)
weather_model = load_keras_model(WEATHER_MODEL_PATH)

def detect_emotion_from_bgr(frame_bgr: np.ndarray) -> str:
    if emotion_model is None:
        return "neutral"
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    x = np.expand_dims(gray[..., np.newaxis], axis=0)
    preds = emotion_model.predict(x, verbose=0)[0]
    return EMOTION_LABELS[int(np.argmax(preds))]

# ===================== APP PAGES =====================
init_db()
if "page" not in st.session_state:
    st.session_state.page = "splash"
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None

# -------- Splash Page --------
if st.session_state.page == "splash":
    st.markdown("<h1 style='text-align:center;'>Emotunes</h1>", unsafe_allow_html=True)
    time.sleep(2.5)
    st.session_state.page = "auth"
    st.experimental_rerun()

# -------- Auth Page --------
elif st.session_state.page == "auth":
    st.markdown("<h1 style='text-align:center;'>Welcome to Emotunes</h1>", unsafe_allow_html=True)
    choice = st.radio("Choose an option", ["Login", "Signup"], horizontal=True)

    if choice == "Signup":
        email = st.text_input("Email")
        password = st.text_input("Create Password", type="password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Send OTP"):
                norm_email = email.strip().lower()
                if get_user_by_email(norm_email):
                    st.error("Email already exists. Please login.")
                else:
                    send_otp(norm_email, purpose="signup")
                    st.session_state.pending_email = norm_email

        with col2:
            if st.button("Resend OTP"):
                if "pending_email" in st.session_state:
                    send_otp(st.session_state.pending_email, purpose="signup")
                else:
                    st.info("Enter your email and click Send OTP first.")

        otp = st.text_input("Enter OTP").strip()

        if st.button("Verify & Signup", use_container_width=True):
            target_email = st.session_state.get("pending_email")
            if not target_email:
                st.error("Please request OTP first.")
            elif not verify_otp(target_email, otp, purpose="signup"):
                st.error("Invalid or expired OTP.")
            else:
                try:
                    create_user(target_email, hash_password(password))
                    st.success("Signup successful 🎉")
                    st.session_state.email = target_email
                    st.session_state.page = "userinfo"
                    st.experimental_rerun()
                except sqlite3.IntegrityError:
                    st.error("Email already exists. Please login.")
