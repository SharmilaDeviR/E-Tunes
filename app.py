# app.py — Emotunes: Hear how you feel
# Python 3.10 / TensorFlow 2.13

import os
import io
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
MODELS_DIR = "models"
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.h5")
WEATHER_MODEL_PATH = os.path.join(MODELS_DIR, "weather_model.h5")  # optional
OTP_TTL_SEC = 300  # 5 minutes

# Load .env (create a .env file in this folder)
# SPOTIPY_CLIENT_ID=...
# SPOTIPY_CLIENT_SECRET=...
# SMTP_HOST=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USER=your@gmail.com
# SMTP_PASS=your_app_password
# SMTP_FROM=Emotunes <your@gmail.com>
# SMTP_TLS=true
load_dotenv()

# ===================== STREAMLIT SETUP =====================
st.set_page_config(page_title="Emotunes", layout="centered", page_icon="🎧")

# Ensure OTP cache is always available
if "otp_cache" not in st.session_state:
    st.session_state.otp_cache = {}

# ---- Background image (themebg.png) applied to entire app ----
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
    /* slight glass effect for main blocks */
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

def load_keras_model(path: str):
    try:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)
    except Exception as e:
        st.warning(f"Could not load model at {path}: {e}")
    return None

emotion_model = load_keras_model(EMOTION_MODEL_PATH)
weather_model = load_keras_model(WEATHER_MODEL_PATH)  # optional, not used below

def detect_emotion_from_bgr(frame_bgr: np.ndarray) -> str:
    """Preprocess to match model input: (224,224,3) RGB, normalized [0,1]."""
    if emotion_model is None:
        return "neutral"
    # Convert BGR->RGB and resize to 224x224
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    x = np.expand_dims(frame_resized, axis=0)  # (1,224,224,3)
    preds = emotion_model.predict(x, verbose=0)[0]
    return EMOTION_LABELS[int(np.argmax(preds))]

# ===================== CONTEXT / RECOMMENDER =====================
def daypart_from_now(now: Optional[dt.datetime] = None) -> str:
    now = now or dt.datetime.now()
    h = now.hour
    if 5 <= h < 8: return "early_morning"
    if 8 <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 22: return "evening"
    return "late_night"

def mood_to_tag(emotion: str, weather: str, daypart: str) -> str:
    e = (emotion or "neutral").lower()
    w = (weather or "clear").lower()
    d = daypart

    base = {
        "happy":"happy",
        "surprise":"energetic",
        "neutral":"chill",
        "sad":"sad",
        "fear":"focus",
        "disgust":"focus",
        "angry":"energetic",
    }.get(e, "chill")

    if "rain" in w or "storm" in w:
        if base in ("happy","energetic"): base = "chill"
    if "clear" in w and d in ("morning","afternoon"):
        if base == "chill": base = "happy"
    if d == "late_night" and base in ("energetic","happy"):
        base = "chill"
    return base

CURATED_PLAYLISTS = {
    "happy": ["37i9dQZF1DXdPec7aLTmlC", "37i9dQZF1DWTwbZHrJRIgD"],
    "sad": ["37i9dQZF1DX7qK8ma5wgG1", "37i9dQZF1DX7gIoKXt0gmx"],
    "chill": ["37i9dQZF1DX4WYpdgoIcn6", "37i9dQZF1DX4sWSpwq3LiO"],
    "energetic": ["37i9dQZF1DX8FwnYE6PRvL", "37i9dQZF1DX8tZsk68tuDw"],
    "focus": ["37i9dQZF1DX3PFzdbtx1Us", "37i9dQZF1DX8Uebhn9wzrS"],
    # language buckets (can expand)
    "tamil": ["37i9dQZF1DX2n4gU7dUeAZ"],
    "telugu": ["37i9dQZF1DX5Ejj0EkURtP"],
    "hindi": ["37i9dQZF1DX1i3hvzHpcQV"],
    "english": ["37i9dQZF1DXcBWIGoYBM5M"],
    "spanish": ["37i9dQZF1DX10zKzsJ2jva"],
}

def normalize_language(lang: str) -> str:
    s = (lang or "").strip().lower()
    aliases = {"en":"english","hi":"hindi","ta":"tamil","te":"telugu","es":"spanish"}
    return aliases.get(s, s)

def build_queries(tag: str, languages: List[str]) -> List[str]:
    langs = [normalize_language(l) for l in languages]
    q = [f"{tag} hits"]
    for lang in langs[:3]:
        q += [f"{lang} {tag}", f"{lang} {tag} playlist"]
    return q

# ===================== SPOTIFY (Client Credentials) =====================
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
    st.warning("⚠️ Spotify credentials missing in .env (SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET). Searching may fail.")
try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET
    ))
except Exception as e:
    sp = None
    st.warning(f"Spotify init failed: {e}")

def search_playlists(query: str, limit: int = 8) -> List[Dict]:
    if sp is None:
        return []
    try:
        res = sp.search(q=query, type="playlist", limit=limit)
    except Exception as e:
        st.warning(f"Spotify search failed for '{query}': {e}")
        return []

    items = res.get("playlists", {}).get("items", []) or []
    results = []
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = it.get("id")
        if not pid:
            continue
        results.append({
            "name": it.get("name", "Untitled"),
            "owner": (it.get("owner") or {}).get("display_name", "Unknown"),
            "id": pid,
            "url": f"https://open.spotify.com/playlist/{pid}",
            "image": (it.get("images") or [{}])[0].get("url")
        })
    return results

def recommend_playlists(emotion: str, weather: str, languages_csv: str) -> List[Dict]:
    dpart = daypart_from_now()
    tag = mood_to_tag(emotion, weather, dpart)
    langs = [normalize_language(x) for x in (languages_csv or "").split(",") if x]

    # curated first
    picks: List[Dict] = []
    for key in [tag] + langs:
        ids = CURATED_PLAYLISTS.get(key, [])
        for pid in ids:
            picks.append({"name": f"Curated: {key.title()}", "id": pid,
                          "url": f"https://open.spotify.com/playlist/{pid}", "image": None})

    # dynamic search fallback
    for q in build_queries(tag, langs or ["english"]):
        picks.extend(search_playlists(q, limit=4))

    # dedupe by id
    seen, unique = set(), []
    for p in picks:
        pid = p.get("id")
        if pid and pid not in seen:
            unique.append(p)
            seen.add(pid)
    return unique[:12]

# ===================== UI HELPERS =====================
def big_center_title(text: str):
    st.markdown(
        f"""
        <h1 style="text-align:center; font-size:56px; margin: 2rem 0 0.5rem 0;">
            {text}
        </h1>
        """, unsafe_allow_html=True
    )

def subtle_caption(text: str):
    st.markdown(f"<p style='text-align:center; opacity:0.75;'>{text}</p>", unsafe_allow_html=True)

def feedback_fab():
    # floating feedback button (bottom-right)
    st.markdown("""
        <style>
        .fab { position: fixed; right: 24px; bottom: 24px; z-index: 9999; }
        </style>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='fab'></div>", unsafe_allow_html=True)
        with st.expander("💬 Feedback"):
            user = get_user_by_email(st.session_state.get("email","")) if "email" in st.session_state else None
            fb = st.text_area("Tell us what you think:", placeholder="Your feedback helps improve Emotunes ❤️")
            if st.button("Submit", use_container_width=True):
                if user:
                    add_feedback(user["id"], fb or "")
                    st.success("Thanks! Feedback submitted.")
                else:
                    st.warning("Please login first to submit feedback.")

# ===================== APP PAGES =====================
init_db()
if "page" not in st.session_state:
    st.session_state.page = "splash"
if "captured_img" not in st.session_state:
    st.session_state.captured_img = None
if "camera_consent" not in st.session_state:
    st.session_state.camera_consent = False

# -------- Splash Page --------
if st.session_state.page == "splash":
    big_center_title("Emotunes — Hear how you feel")
    subtle_caption("loading your personalized music experience…")
    time.sleep(2)
    st.session_state.page = "auth"
    st.rerun()

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
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.error("Email already exists. Please login.")

    else:  # Login
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Login"):
                user = get_user_by_email(email)
                if not user:
                    st.error("No account found. Please signup.")
                elif not verify_password(password, user["password_hash"]):
                    st.error("Incorrect password.")
                else:
                    st.session_state.email = email.strip().lower()
                    st.success("Logged in ✅")
                    st.session_state.page = "userinfo"
                    st.rerun()
        with c2:
            if st.button("Send OTP (Login)"):
                if not email.strip():
                    st.error("Enter your email first.")
                elif not get_user_by_email(email):
                    st.error("Email not found. Please signup.")
                else:
                    send_otp(email, purpose="login")
                    st.session_state.login_email = email.strip().lower()
        with c3:
            otp_login = st.text_input("OTP (for Login)").strip()
            if st.button("Verify OTP Login"):
                target = st.session_state.get("login_email")
                if not target:
                    st.error("Request OTP first.")
                elif verify_otp(target, otp_login, purpose="login"):
                    st.session_state.email = target
                    st.success("Logged in via OTP ✅")
                    st.session_state.page = "userinfo"
                    st.rerun()
                else:
                    st.error("Invalid/expired OTP.")

# -------- User Info Page --------
elif st.session_state.page == "userinfo":
    big_center_title("Tell us about you")
    st.info("Give **true** information to improve your experience with Emotunes.")
    age = st.number_input("Age", min_value=10, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    languages = st.multiselect("Preferred languages (choose at least 2)",
                               ["English","Hindi","Tamil","Telugu","Spanish"])

    # Ethical camera usage: explicit consent and optional capture UI
    st.session_state.camera_consent = st.checkbox(
        "I consent to using my webcam for one-time emotion analysis (optional).",
        value=st.session_state.camera_consent
    )
    img = None
    if st.session_state.camera_consent:
        img = st.camera_input("Allow camera access (we will only analyze locally and won't store your image).")

    can_continue = (len(languages) >= 2)
    if len(languages) < 2:
        st.warning("Please select at least 2 languages.")
    if st.button("Continue ▶", type="primary", disabled=not can_continue, use_container_width=True):
        update_user_profile(st.session_state.email, int(age), gender, ",".join(languages))
        st.session_state.captured_img = img.getvalue() if (st.session_state.camera_consent and img) else None
        st.session_state.page = "music"
        st.rerun()

# -------- Music Page --------
elif st.session_state.page == "music":
    big_center_title("🎶 Your Emotunes")
    user = get_user_by_email(st.session_state.get("email",""))
    if not user:
        st.warning("Please login again.")
        st.session_state.page = "auth"
        st.rerun()

    # Use prior consented capture if available; allow optional re-capture
    img_bytes = st.session_state.get("captured_img")
    if st.session_state.camera_consent:
        recapture = st.checkbox("Re-capture emotion (optional)", value=False)
        if recapture:
            img_file = st.camera_input("Capture for emotion analysis")
            if img_file:
                img_bytes = img_file.getvalue()
                st.session_state.captured_img = img_bytes

    # Convert to OpenCV BGR and detect
    if img_bytes:
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        detected_emotion = detect_emotion_from_bgr(frame_bgr)
        st.success(f"Detected emotion: **{detected_emotion.title()}**")
    else:
        detected_emotion = "neutral"
        st.info("No camera image provided. Defaulting to **Neutral** (you can select manually).")

    # Manual override
    manual = st.selectbox(
        "Pick a mood manually (optional)",
        EMOTION_LABELS,
        index=EMOTION_LABELS.index(detected_emotion) if detected_emotion in EMOTION_LABELS else 4
    )
    emotion_used = manual or detected_emotion

    # Weather (placeholder / TODO: connect your weather model)
    weather = "clear"

    # Recommend playlists
    playlists = recommend_playlists(emotion_used, weather, user["languages"] or "English,Hindi")
    if not playlists:
        st.error("Couldn’t fetch playlists. Check Spotify credentials in .env.")
    else:
        st.subheader("Recommended Playlists")
        for p in playlists:
            c1, c2 = st.columns([1,3], vertical_alignment="center")
            with c1:
                if p.get("image"):
                    st.image(p["image"])
                else:
                    st.write("🎵")
            with c2:
                st.markdown(f"**{p['name']}**")
                st.markdown(f"[Open in Spotify]({p['url']})")

    st.divider()
    feedback_fab()
