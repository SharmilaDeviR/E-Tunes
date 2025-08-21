"""
spotify_api.py — Spotify integration utilities
- Uses spotipy for Web API
- Returns playlist and track links (no playback control)
"""

from __future__ import annotations
from typing import List, Dict, Optional
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load .env
load_dotenv()

SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")

if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
    raise ValueError("❌ Spotify credentials missing in .env")

# Global client (app-only auth)
auth_manager = SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
)
sp = spotipy.Spotify(auth_manager=auth_manager)

# ---------- Playlist Search ----------
def search_playlists(query: str, limit: int = 10) -> List[Dict]:
    res = sp.search(q=query, type="playlist", limit=limit)
    items = res.get("playlists", {}).get("items", [])
    return [
        {
            "name": it["name"],
            "owner": it["owner"].get("display_name", "Unknown"),
            "id": it["id"],
            "url": f"https://open.spotify.com/playlist/{it['id']}",
            "image": (it["images"][0]["url"] if it.get("images") else None),
        }
        for it in items
    ]

# ---------- Playlist Tracks ----------
def get_playlist_tracks(playlist_id: str, limit: int = 50) -> List[Dict]:
    res = sp.playlist_items(playlist_id, additional_types=("track",), limit=limit)
    items = res.get("items", [])
    out = []
    for it in items:
        tr = it.get("track")
        if not tr:
            continue
        out.append({
            "name": tr["name"],
            "artists": ", ".join(a["name"] for a in tr.get("artists", [])),
            "id": tr["id"],
            "url": f"https://open.spotify.com/track/{tr['id']}" if tr.get("id") else tr.get("external_urls", {}).get("spotify"),
            "preview_url": tr.get("preview_url"),
        })
    return out

# ---------- Curated Mood Playlists ----------
CURATED_PLAYLISTS = {
    # Universal moods
    "happy": [
        "37i9dQZF1DXdPec7aLTmlC",  # Happy Hits!
        "37i9dQZF1DWTwbZHrJRIgD",  # Good Vibes
    ],
    "sad": [
        "37i9dQZF1DX7qK8ma5wgG1",  # Life Sucks
        "37i9dQZF1DX7gIoKXt0gmx",  # Deep Dark Indie
    ],
    "chill": [
        "37i9dQZF1DX4WYpdgoIcn6",
        "37i9dQZF1DX4sWSpwq3LiO",
    ],
    "energetic": [
        "37i9dQZF1DX8FwnYE6PRvL",  # Power Workout
        "37i9dQZF1DX8tZsk68tuDw",  # Beast Mode
    ],
    "focus": [
        "37i9dQZF1DX3PFzdbtx1Us",  # Lo-Fi Beats
        "37i9dQZF1DX8Uebhn9wzrS",  # Deep Focus
    ],
    # Indian languages samples
    "tamil": ["37i9dQZF1DX2n4gU7dUeAZ", "37i9dQZF1DXaS1OQhH1UqO"],   # Tamil Hits
    "telugu": ["37i9dQZF1DX5Ejj0EkURtP", "37i9dQZF1DXbvd62nQzFJ1"],  # Telugu
    "hindi": ["37i9dQZF1DX1i3hvzHpcQV", "37i9dQZF1DX4dyzvuaRJ0n"],   # Hindi
    "english": ["37i9dQZF1DXcBWIGoYBM5M"],
    "spanish": ["37i9dQZF1DX10zKzsJ2jva"],
}

def pick_curated_playlist(keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in CURATED_PLAYLISTS and CURATED_PLAYLISTS[k]:
            return CURATED_PLAYLISTS[k][0]
    return None
