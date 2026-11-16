from src.recommender import Song, UserProfile, Recommender

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# --- Scoring helpers used by the Streamlit GUI ---------------------------------

from src.recommender import DEFAULT_WEIGHTS, max_score, score_breakdown, score_song, recommend_songs

POP_FAN = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
SONG = {"genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2}


def test_breakdown_parts_add_up_to_score():
    parts, reasons = score_breakdown(POP_FAN, SONG)
    score, _ = score_song(POP_FAN, SONG)
    assert abs(sum(parts.values()) - score) < 1e-9
    assert len(reasons) == 4


def test_max_score_matches_default_weights():
    assert max_score() == sum(DEFAULT_WEIGHTS.values()) == 6.0


def test_custom_weights_change_points():
    parts, _ = score_breakdown(POP_FAN, SONG, {"genre": 3.0, "mood": 0.0})
    assert parts["genre"] == 3.0
    assert parts["mood"] == 0.0
    assert max_score({"genre": 3.0, "mood": 0.0}) == 6.5


def test_related_genre_earns_half_the_genre_weight():
    parts, _ = score_breakdown(POP_FAN, {**SONG, "genre": "indie pop"}, {"genre": 2.0})
    assert parts["genre"] == 1.0


def test_recommend_songs_accepts_weights():
    songs = [
        {"title": "A", **SONG},
        {"title": "B", "genre": "rock", "mood": "happy", "energy": 0.8, "acousticness": 0.2},
    ]
    # With genre weight zero, the two songs tie and catalog order is kept.
    ranked = recommend_songs(POP_FAN, songs, k=2, weights={"genre": 0.0})
    assert ranked[0][1] == ranked[1][1]
    assert [s["title"] for s, _, _ in ranked] == ["A", "B"]


# --- Catalog integrity ------------------------------------------------------------

import csv
import re
from pathlib import Path

CATALOG = Path(__file__).resolve().parent.parent / "data" / "songs.csv"


def _normalized_key(title: str, artist: str) -> str:
    t = re.sub(r"\s*[\(\[].*?[\)\]]", "", title.lower())
    t = re.sub(r"\s+-\s+.*$", "", t)
    t = re.sub(r"[^a-z0-9 ]", "", t).strip()
    return f"{t}|{artist.split(',')[0].strip().lower()}"


def test_catalog_has_no_duplicate_songs():
    with open(CATALOG, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids = [r["spotify_id"] for r in rows]
    keys = [_normalized_key(r["title"], r["artist"]) for r in rows]
    assert len(ids) == len(set(ids)), "duplicate Spotify track ids"
    assert len(keys) == len(set(keys)), "same song (title + lead artist) listed twice"


def test_catalog_features_are_in_range():
    with open(CATALOG, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for field in ("energy", "valence", "danceability", "acousticness"):
                assert 0.0 <= float(row[field]) <= 1.0, (row["title"], field)
            assert row["spotify_id"], row["title"]
