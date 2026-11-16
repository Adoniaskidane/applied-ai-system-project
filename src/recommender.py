import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

# Genre pairs that earn partial credit (+0.5) instead of a full match.
RELATED_GENRES = {
    frozenset(("pop", "indie pop")),
    frozenset(("lofi", "ambient")),
}

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    spotify_id: str = ""  # Spotify track id, used by the GUI to embed a player

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Rank songs with the same scoring rules as score_song and return the top k."""
        prefs = asdict(user)
        ranked = sorted(self.songs, key=lambda s: score_song(prefs, asdict(s))[0], reverse=True)
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Plain-language reasons this song earned its score."""
        score, reasons = score_song(asdict(user), asdict(song))
        return f"Score {score:.2f}: " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Read songs.csv into a list of dicts, converting numeric columns to int/float."""
    int_fields = {"id"}
    float_fields = {"energy", "tempo_bpm", "valence", "danceability", "acousticness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs

DEFAULT_WEIGHTS = {
    "genre": 1.0,     # EXPERIMENT: halved from 2.0 (related genre earns half)
    "mood": 1.5,
    "energy": 3.0,    # EXPERIMENT: doubled from 1.5
    "acoustic": 0.5,
}


def max_score(weights: Optional[Dict] = None) -> float:
    """Best possible score under the given weights (every rule fully satisfied)."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return sum(w.values())


def score_breakdown(user_prefs: Dict, song: Dict, weights: Optional[Dict] = None) -> Tuple[Dict[str, float], List[str]]:
    """Return per-feature points ({genre, mood, energy, acoustic}) and the matching reasons."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    parts = {"genre": 0.0, "mood": 0.0, "energy": 0.0, "acoustic": 0.0}
    reasons = []

    user_genre = user_prefs["favorite_genre"].strip().lower()
    song_genre = song["genre"].strip().lower()
    if song_genre == user_genre:
        parts["genre"] = w["genre"]
        reasons.append(f"genre match: {song_genre} (+{parts['genre']:.1f})")
    elif frozenset((user_genre, song_genre)) in RELATED_GENRES:
        parts["genre"] = w["genre"] / 2
        reasons.append(f"related genre: {song_genre} (+{parts['genre']:.1f})")

    user_mood = user_prefs["favorite_mood"].strip().lower()
    song_mood = song["mood"].strip().lower()
    if song_mood == user_mood:
        parts["mood"] = w["mood"]
        reasons.append(f"mood match: {song_mood} (+{parts['mood']:.1f})")

    # Closeness, not size: full points when energy equals the target.
    parts["energy"] = w["energy"] * (1 - abs(song["energy"] - user_prefs["target_energy"]))
    reasons.append(f"energy {song['energy']:.2f} vs target {user_prefs['target_energy']:.2f} (+{parts['energy']:.2f})")

    if user_prefs["likes_acoustic"]:
        parts["acoustic"] = w["acoustic"] * song["acousticness"]
        reasons.append(f"acoustic sound (+{parts['acoustic']:.2f})")
    else:
        parts["acoustic"] = w["acoustic"] * (1 - song["acousticness"])
        reasons.append(f"non-acoustic sound (+{parts['acoustic']:.2f})")

    return parts, reasons


def score_song(user_prefs: Dict, song: Dict, weights: Optional[Dict] = None) -> Tuple[float, List[str]]:
    """Score one song against user prefs; return (score, reasons) per the Algorithm Recipe."""
    parts, reasons = score_breakdown(user_prefs, song, weights)
    # Plain left-to-right addition (not sum(), which uses compensated
    # summation on 3.12+) keeps scores identical to the original recipe.
    score = 0.0
    for points in parts.values():
        score += points
    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, weights: Optional[Dict] = None) -> List[Tuple[Dict, float, str]]:
    """Score every song, sort highest first, and return the top k as (song, score, explanation)."""
    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, weights)
        scored.append((song, score, "; ".join(reasons)))

    return sorted(scored, key=lambda item: item[1], reverse=True)[:k]
