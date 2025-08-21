"""
recommender.py — combine emotion + weather + time + user profile to choose Spotify playlists
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import datetime as _dt

from spotify_api import search_playlists, pick_curated_playlist

Daypart = str  # "early_morning"|"morning"|"afternoon"|"evening"|"late_night"

@dataclass
class Profile:
    age: Optional[int]
    gender: Optional[str]
    languages: List[str]  # ["English","Tamil",...]

def daypart_from_now(now: Optional[_dt.datetime] = None) -> Daypart:
    now = now or _dt.datetime.now()
    h = now.hour
    if 5 <= h < 8: return "early_morning"
    if 8 <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 22: return "evening"
    return "late_night"

def normalize_language(lang: str) -> str:
    s = lang.strip().lower()
    aliases = {
        "en":"english","english":"english",
        "hi":"hindi","hin":"hindi","hindi":"hindi",
        "ta":"tamil","tamil":"tamil",
        "te":"telugu","telugu":"telugu",
        "es":"spanish","spanish":"spanish"
    }
    return aliases.get(s, s)

def mood_to_tags(emotion: str, weather: str, daypart: Daypart) -> List[str]:
    """
    Map signals -> abstract tags (happy/chill/energetic/focus/sad)
    """
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

    # Weather modulation
    if "rain" in w or "storm" in w:
        base = "chill" if base in ("happy","energetic") else base
    if "clear" in w and d in ("morning","afternoon"):
        base = "happy" if base == "chill" else base
    if d in ("late_night",) and base in ("energetic","happy"):
        base = "chill"

    return [base]

def build_queries(tags: List[str], languages: List[str]) -> List[str]:
    langs = [normalize_language(l) for l in languages]
    queries = []
    for tag in tags:
        # generic
        queries.append(f"{tag} hits")
        # multilingual
        for lang in langs[:3]:  # cap to 3 to keep it tight
            queries.append(f"{lang} {tag}")
            queries.append(f"{lang} {tag} playlist")
    return queries

def recommend_playlists(
    *, emotion: str, weather: str, profile: Profile, now: Optional[_dt.datetime] = None
) -> List[Dict]:
    """
    Return a ranked list of playlists (dicts with name/url/id/image).
    Strategy:
      1) try curated by mood/language
      2) fallback to Spotify search queries
    """
    dpart = daypart_from_now(now)
    tags = mood_to_tags(emotion, weather, dpart)

    # 1) curated pick by mood + first language preference
    lang_keys = [normalize_language(l) for l in (profile.languages or [])]
    curated_keys = tags + lang_keys
    curated_id = pick_curated_playlist(curated_keys)
    results: List[Dict] = []
    if curated_id:
        results.append({
            "name": f"Curated for {tags[0].title()}",
            "id": curated_id,
            "url": f"https://open.spotify.com/playlist/{curated_id}",
            "image": None
        })

    # 2) dynamic search
    for q in build_queries(tags, lang_keys or ["english"]):
        for item in search_playlists(q, limit=4):
            results.append(item)

    # dedupe by playlist id
    seen = set()
    unique = []
    for r in results:
        pid = r.get("id") or r.get("url")
        if pid in seen:
            continue
        seen.add(pid)
        unique.append(r)

    return unique[:15]  # keep it short for UI
