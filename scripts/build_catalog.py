"""
Build data/songs.csv from the public Spotify Tracks Dataset (~114k real tracks).

Every song in the catalog is a real track. Its Spotify track id and audio
features (energy, tempo, valence, danceability, acousticness) come straight from
the dataset, so the GUI can embed a Spotify player for it.

The catalog is built in two parts:
  1. 19 curated "slots" (ids 1-19): one per original genre/mood slot, each either a
     hand-picked well-known song or an auto-picked closest match. Moods set by hand.
  2. Bulk fill (ids 20+): the most popular tracks in each of our 16 genres, up to
     --per-genre songs each. Their mood is inferred (see infer_mood).

No duplicates: the dataset lists many songs more than once (album re-releases,
remasters, the same track under several genres). Songs are de-duplicated by
Spotify track id AND by normalized title + lead artist.

English only: songs with vocals must be in English (see is_non_english). Instrumental
tracks (Spotify instrumentalness >= 0.5) have no lyrics, so they are kept; this is
what keeps lofi, ambient and classical in the catalog.

Only tracks the dataset files under a single genre are used for bulk fill, because
its genre labels come from playlists and are noisy (a pop hit can be filed under
"rock").

Usage (from the project root):
    python scripts/build_catalog.py                 # downloads the dataset if needed (~20 MB)
    python scripts/build_catalog.py --source path/to/dataset.csv
    python scripts/build_catalog.py --per-genre 100     # bigger catalog

Dataset: https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset
(Spotify Web API audio features, collected by Maharshi Pandya; also on Kaggle.)

Note: the dataset has no "mood" column. Curated moods are hand-assigned; bulk-fill
moods are inferred from audio features, so treat them as approximate.
"""

import argparse
import csv
import re
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
DEFAULT_CACHE = ROOT / ".cache" / "spotify_tracks.csv"
OUTPUT = ROOT / "data" / "songs.csv"

MIN_POPULARITY = 45  # curated auto-pick skips obscure tracks so the catalog feels recognizable
BULK_MIN_POPULARITY = 30  # bulk fill goes a little deeper to reach enough songs per genre
DEFAULT_PER_GENRE = 70

# Our genre name -> genre label used in the Spotify dataset (for auto-pick).
GENRE_MAP = {
    "pop": "pop", "indie pop": "indie-pop", "lofi": "study", "ambient": "ambient",
    "rock": "rock", "metal": "metal", "jazz": "jazz", "synthwave": "synth-pop",
    "country": "country", "hip hop": "hip-hop", "r&b": "r-n-b", "edm": "edm",
    "classical": "classical", "reggae": "reggae", "blues": "blues",
    "folk": "folk",
}

# (genre, mood, pick, auto-pick target: energy, tempo, valence, danceability, acousticness)
# The original catalog's genre/mood slots (3 lofi, 2 pop, 1 of most others), minus latin,
# which was removed from the catalog.
SLOTS = [
    ("pop", "happy", ("Levitating", "Dua Lipa"), None),
    ("lofi", "chill", None, (0.42, 78, 0.56, 0.62, 0.71)),
    ("rock", "intense", ("Smells Like Teen Spirit", "Nirvana"), None),
    ("lofi", "chill", None, (0.35, 72, 0.60, 0.58, 0.86)),
    ("pop", "intense", ("Blinding Lights", "The Weeknd"), None),
    ("ambient", "chill", ("An Ending (Ascent)", "Brian Eno"), None),
    ("jazz", "relaxed", ("Take Five", "Dave Brubeck"), None),
    ("synthwave", "moody", ("Nightcall", "Kavinsky"), None),
    ("lofi", "focused", None, (0.40, 80, 0.59, 0.60, 0.78)),
    ("indie pop", "happy", ("Young Folks", "Peter Bjorn and John"), None),
    ("country", "nostalgic", ("Take Me Home, Country Roads", "John Denver"), None),
    ("hip hop", "confident", ("HUMBLE.", "Kendrick Lamar"), None),
    ("r&b", "romantic", ("Thinkin Bout You", "Frank Ocean"), None),
    ("edm", "euphoric", ("Levels", "Avicii"), None),
    ("classical", "melancholic", ("Clair de Lune", "Debussy"), None),
    ("reggae", "uplifting", ("Three Little Birds", "Bob Marley"), None),
    ("metal", "aggressive", ("Duality", "Slipknot"), None),
    ("blues", "sad", ("The Thrill Is Gone", "B.B. King"), None),
    ("folk", "hopeful", ("Ho Hey", "The Lumineers"), None),
]

# --- English-only filter --------------------------------------------------------------
INSTRUMENTAL_THRESHOLD = 0.5

# Dataset genres that are clearly non-English-language scenes. An artist whose tracks are
# mostly filed under these is treated as singing in another language. (french, german and
# swedish are deliberately NOT here: English-singing acts like M83 and Avicii are filed there.)
NON_ENGLISH_GENRES = {
    "anime", "brazil", "cantopop", "forro", "indian", "iranian", "j-dance", "j-idol", "j-pop",
    "j-rock", "k-pop", "latin", "latino", "malay", "mandopop", "mpb", "pagode", "pop-film",
    "reggaeton", "salsa", "samba", "sertanejo", "spanish", "tango", "turkish",
}

# Common function words in Spanish, Portuguese, French, Italian and German that rarely
# appear in English titles. One hit in a title marks the song as non-English. Short words that
# also occur in English titles ("ho", "ya", "lo", "um", "y", "solo") are deliberately left out.
FOREIGN_TITLE_WORDS = {
    # Spanish / Portuguese
    "el", "los", "las", "del", "que", "mi", "tu", "te", "yo", "por", "para", "con", "sin",
    "amor", "corazon", "eres", "quiero", "como", "cuando", "nao", "voce", "eu", "uma",
    "meu", "minha", "seu", "sua", "pra", "mas", "muito", "nada", "todo", "vida", "noche",
    "una", "es", "se", "su", "mas", "tus", "mis", "sus", "nos", "ni", "pero",
    # French
    "le", "les", "des", "et", "je", "moi", "toi", "une", "est", "pas", "vie", "en", "au",
    "aux", "avec", "pour", "dans", "sur", "qui", "mon", "ma", "ton", "ta", "toujours", "rien",
    # Italian
    "il", "di", "che", "ti", "amore", "sono", "non", "della", "nel", "per", "ancora",
    # German
    "der", "das", "und", "ich", "nicht", "ein", "eine", "mit", "ist", "dich", "mich", "liebe",
    "auf", "wir", "ihr", "zu", "nur", "noch",
    # Portuguese extras
    "ao", "vivo", "acustico", "nos", "sem", "voar", "reino", "dia", "noite", "votos", "arvore",
    # Indonesian / Malay
    "aku", "kau", "dan", "yang", "kita", "cinta", "jarak", "baruku", "apa", "adanya", "gue",
    # Hindi / Punjabi / Tamil transliteration
    "dil", "tera", "teri", "mera", "meri", "mere", "tum", "hai", "nahi", "ka", "ki", "ke",
    "jaan", "ishq", "pyaar", "yaar", "sajna", "nee", "kanave", "naan", "pazham",
}


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _clean_title(title: str) -> str:
    title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title)  # drop "(feat. X)", "[Remastered]"
    return re.sub(r"\s+-\s+.*$", "", title)            # drop " - Radio Edit", " - Acustico"


def title_looks_non_english(title: str) -> bool:
    core = _clean_title(title)
    # Any letter outside the Latin alphabet (CJK, Devanagari, Arabic, Cyrillic, Hangul, ...).
    if any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in core):
        return True
    words = re.findall(r"[a-z]+", _strip_accents(core).lower())
    return any(w in FOREIGN_TITLE_WORDS for w in words)


# Dataset genres used as examples of English-language titles, for the vocabulary check.
ENGLISH_EXAMPLE_GENRES = {
    "country", "rock", "hard-rock", "punk", "punk-rock", "emo", "grunge", "british", "singer-songwriter",
    "songwriter", "alt-rock", "alternative", "indie", "soul", "gospel", "bluegrass", "honky-tonk",
    "rock-n-roll", "rockabilly", "power-pop", "show-tunes", "disney", "metalcore", "psych-rock", "folk", "blues",
}
# Same idea for the foreign side; french/german/swedish are fine here because this only
# learns vocabulary, it doesn't flag artists by genre.
FOREIGN_EXAMPLE_GENRES = NON_ENGLISH_GENRES | {"french", "german", "swedish"}

# Artists whose vocals aren't in English but whose titles look English (or are too few to judge),
# found by reviewing the catalog. Mostly Punjabi, Hindi and Tamil acts with English-language titles.
EXCLUDE_ARTISTS = {
    "anirudh ravichander", "shubh", "ap dhillon", "mc stan", "badshah", "sachin-jigar", "shankar-ehsaan-loy",
    "harish raghavan", "king", "paradox", "aleemrk", "dhyo haw", "jukka poika", "bizarrap", "måneskin",
    "synapson", "vieux farka touré", "golden ganga", "bubaseta", "edson gomes", "gabriel elias",
    "chimarruts", "el kuelgue", "stailok", "shamanes crew", "enur", "morada", "7 minutoz", "meruem",
    "anirap", "enygma rapper", "mhrap", "los hermanos", "charlie brown jr.", "kid abelha", "vtornik",
    "agnes nunes", "ami mishra", "amrit maan", "baby do brasil", "blitz", "consciência humana",
    "dhibu ninan thomas", "diego herrera", "divine", "dj jamaika", "dr. silvana & cia.",
    "el chapo de sinaloa", "el coyote y su banda tierra santa", "el mimoso luis antonio lópez",
    "emiway bantai", "facção central", "guru randhawa", "harnoor", "herva doce", "imran khan",
    "indochine", "inimigos do rei", "jaura phagwara", "javed-mohsin",
    "joão penca e seus miquinhos amestrados", "karakoram", "karan aujla", "la adictiva",
    "lehmber hussainpuri", "luis arcaraz y su orquesta", "m. s. subbulakshmi", "mahesh raghvan",
    "malladi brothers", "mamonas assassinas", "manik varma", "mtk", "pandit jasraj", "pav dharia",
    "pratsofficial", "prem dhillon", "priya saraiya", "priya sisters", "radha jayalakshmi", "raftaar",
    "rahul deshpande", "rattan mohan sharma", "rav aulakh", "rodrigo zin", "sachet-parampara", "sanam",
    "satinder sartaaj", "saul el jaguar alarcón", "shasings", "sidhu moose wala", "sikkil gurucharan",
    "soolamangalam sisters", "tegi pannu", "vasantrao deshpande", "vedavathi prabhakar",
    "veena sahasrabuddhe", "vilen", "yahoo",
    # Second review pass
    "varinder brar", "mc square", "navv inder", "jaz dhami", "hrjxt", "prabh gill", "bobo shashi", "navaan sandhu", "tyson sidhu", "khan bhaini", "shrey singhal", "jerry", "gulab sidhu", "ali azmat", "bickram ghosh", "galneryus", "nightmare",
}


# Individual vocal tracks in another language from otherwise English-language artists.
EXCLUDE_TITLE_PREFIXES = (
    "pa, pa, pa",  # Mozart, The Magic Flute (sung in German)
)


def _title_tokens(title: str) -> list:
    return re.findall(r"[a-z']+", _strip_accents(_clean_title(title)).lower())


def _doc_freq(df: pd.DataFrame) -> tuple:
    counts = {}
    titles = df.drop_duplicates("track_id")["track_name"]
    for title in titles:
        for word in set(_title_tokens(title)):
            counts[word] = counts.get(word, 0) + 1
    return counts, max(len(titles), 1)


def non_english_artists(df: pd.DataFrame) -> set:
    """Lead artists who look non-English, by genre share or by the words in their titles."""
    rows = df[["track_id", "artists", "track_genre", "track_name"]].drop_duplicates(["track_id", "track_genre"])
    rows = rows.assign(lead=rows["artists"].str.split(";").str[0].str.strip().str.lower())

    # 1. Most of the artist's tracks sit in non-English-language genres.
    share = rows["track_genre"].isin(NON_ENGLISH_GENRES).groupby(rows["lead"]).mean()
    flagged = set(share[share >= 0.5].index)

    # 2. Vocabulary: words clearly typical of English-genre titles vs foreign-genre titles.
    en_counts, en_n = _doc_freq(rows[rows["track_genre"].isin(ENGLISH_EXAMPLE_GENRES)])
    fr_counts, fr_n = _doc_freq(rows[rows["track_genre"].isin(FOREIGN_EXAMPLE_GENRES)])
    english_words, foreign_words = set(), set()
    for word in set(en_counts) | set(fr_counts):
        e, f = en_counts.get(word, 0) / en_n, fr_counts.get(word, 0) / fr_n
        if en_counts.get(word, 0) >= 3 and e > 3 * f:
            english_words.add(word)
        elif fr_counts.get(word, 0) >= 3 and f > 3 * e:
            foreign_words.add(word)
    for lead, titles in rows.drop_duplicates("track_id").groupby("lead")["track_name"]:
        tokens = [w for t in titles for w in _title_tokens(t)]
        en_hits = sum(w in english_words for w in tokens)
        fr_hits = sum(w in foreign_words for w in tokens)
        if fr_hits >= 2 and fr_hits > en_hits:  # clear foreign evidence only
            flagged.add(lead)

    return flagged | EXCLUDE_ARTISTS


def english_vocabulary(df: pd.DataFrame) -> set:
    """Broad English title vocabulary: words seen in 2+ English-genre titles that aren't typical
    of foreign-genre titles. Used as positive evidence that a vocal track is in English."""
    rows = df[["track_id", "track_genre", "track_name"]].drop_duplicates(["track_id", "track_genre"])
    en_counts, en_n = _doc_freq(rows[rows["track_genre"].isin(ENGLISH_EXAMPLE_GENRES)])
    fr_counts, fr_n = _doc_freq(rows[rows["track_genre"].isin(FOREIGN_EXAMPLE_GENRES)])
    return {w for w, c in en_counts.items() if c >= 2 and not (fr_counts.get(w, 0) / fr_n > 2 * c / en_n)}


def english_share(titles, vocabulary: set) -> float:
    tokens = [w for t in titles for w in _title_tokens(t) if not w.isdigit()]
    return sum(w in vocabulary for w in tokens) / len(tokens) if tokens else 0.0


def mark_language(df: pd.DataFrame) -> pd.DataFrame:
    """Add an `english_ok` column: True for instrumentals and for vocal tracks that look English."""
    flagged = non_english_artists(df)
    artists = df["artists"].str.lower().str.split(";").map(lambda names: {n.strip() for n in names})
    any_flagged = artists.map(lambda names: bool(names & EXCLUDE_ARTISTS))
    lead = df["artists"].str.split(";").str[0].str.strip().str.lower()
    instrumental = df["instrumentalness"] >= INSTRUMENTAL_THRESHOLD
    excluded_title = df["track_name"].str.lower().str.startswith(EXCLUDE_TITLE_PREFIXES)

    # Positive evidence: most words in the title, and across the artist's titles, are English.
    vocabulary = english_vocabulary(df)
    artist_titles = df.drop_duplicates("track_id").groupby(lead)["track_name"].apply(list)
    artist_share = artist_titles.map(lambda titles: english_share(titles, vocabulary))
    title_share = df["track_name"].map(lambda t: english_share([t], vocabulary))
    lacks_english = (title_share < 0.5) | (lead.map(artist_share).fillna(0) < 0.5)

    non_english = (lead.isin(flagged) | any_flagged | excluded_title | lacks_english
                   | df["track_name"].map(title_looks_non_english))
    return df.assign(english_ok=instrumental | ~non_english)


# What each mood "looks like" in audio features: (energy, valence, danceability, acousticness),
# taken from the original hand-labeled 20-song catalog (averaged where a mood appeared twice).
MOOD_PROTOTYPES = {
    "happy": (0.79, 0.83, 0.81, 0.27),
    "chill": (0.35, 0.60, 0.54, 0.83),
    "intense": (0.92, 0.63, 0.77, 0.08),
    "relaxed": (0.37, 0.71, 0.54, 0.89),
    "moody": (0.75, 0.49, 0.73, 0.22),
    "focused": (0.40, 0.59, 0.60, 0.78),
    "nostalgic": (0.55, 0.68, 0.57, 0.74),
    "confident": (0.78, 0.62, 0.86, 0.12),
    "romantic": (0.48, 0.58, 0.71, 0.40),
    "euphoric": (0.96, 0.88, 0.92, 0.03),
    "melancholic": (0.22, 0.25, 0.18, 0.97),
    "uplifting": (0.60, 0.83, 0.80, 0.48),
    "aggressive": (0.98, 0.21, 0.45, 0.02),
    "sad": (0.38, 0.30, 0.48, 0.82),
    "playful": (0.85, 0.91, 0.89, 0.30),
    "hopeful": (0.33, 0.66, 0.44, 0.93),
}
MOOD_WEIGHTS = (2.0, 2.0, 1.0, 1.0)  # energy and valence carry most of a song's mood

# Feature weights for auto-pick's "closest track": energy and valence matter most for mood fit.
DISTANCE_WEIGHTS = {"energy": 2.0, "valence": 2.0, "acousticness": 1.0, "danceability": 1.0, "tempo_scaled": 0.5}


def load_dataset(source: Path) -> pd.DataFrame:
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading dataset to {source} ...")
        urllib.request.urlretrieve(DATASET_URL, source)
    df = pd.read_csv(source).dropna(subset=["track_id", "track_name", "artists"])
    df["tempo_scaled"] = df["tempo"] / 200.0
    return mark_language(df)


def find_song(df: pd.DataFrame, title: str, artist: str) -> pd.Series:
    """Most popular dataset row whose title starts with `title` and whose artists include `artist`."""
    match = df[
        df["track_name"].str.lower().str.startswith(title.lower())
        & df["artists"].str.contains(artist, case=False, regex=False)
    ]
    if match.empty:
        raise SystemExit(f"Not found in dataset: {title!r} by {artist!r}")
    return match.sort_values("popularity", ascending=False).iloc[0]


def auto_pick(df: pd.DataFrame, genre: str, target: tuple, used_ids: set) -> pd.Series:
    """Popular single-genre track in `genre` whose features are closest to `target`."""
    # The dataset files some tracks under several genres; keep only tracks listed under exactly one.
    genre_counts = df.groupby("track_id")["track_genre"].nunique()
    single = genre_counts[genre_counts == 1].index
    pool = df[
        (df["track_genre"] == GENRE_MAP[genre])
        & df["track_id"].isin(single)
        & (df["popularity"] >= MIN_POPULARITY)
        & df["english_ok"]
        & ~df["track_id"].isin(used_ids)
    ].drop_duplicates("track_id")
    energy, tempo, valence, dance, acoustic = target
    goal = {"energy": energy, "valence": valence, "acousticness": acoustic,
            "danceability": dance, "tempo_scaled": tempo / 200.0}
    distance = sum(w * (pool[f] - goal[f]) ** 2 for f, w in DISTANCE_WEIGHTS.items())
    return pool.loc[distance.idxmin()]


def infer_mood(track: pd.Series) -> str:
    """Mood of the nearest prototype in (energy, valence, danceability, acousticness) space."""
    features = (track["energy"], track["valence"], track["danceability"], track["acousticness"])
    def distance(proto):
        return sum(w * (f - p) ** 2 for w, f, p in zip(MOOD_WEIGHTS, features, proto))
    return min(MOOD_PROTOTYPES, key=lambda mood: distance(MOOD_PROTOTYPES[mood]))


def song_key(title: str, artists: str) -> str:
    """Normalized 'title|lead artist' so re-releases and remasters count as the same song."""
    t = title.lower()
    t = re.sub(r"\s*[\(\[].*?[\)\]]", "", t)  # (feat. X), [Remastered], (Live) ...
    t = re.sub(r"\s+-\s+.*$", "", t)            # " - Radio Edit", " - 2011 Remaster" ...
    t = re.sub(r"[^a-z0-9 ]", "", t).strip()
    lead = artists.split(";")[0].strip().lower()
    return f"{t}|{lead}"


def make_row(song_id: int, track: pd.Series, genre: str, mood: str) -> dict:
    return {
        "id": song_id,
        "title": track["track_name"],
        "artist": track["artists"].replace(";", ", "),
        "genre": genre,
        "mood": mood,
        "energy": round(float(track["energy"]), 2),
        "tempo_bpm": round(float(track["tempo"])),
        "valence": round(float(track["valence"]), 2),
        "danceability": round(float(track["danceability"]), 2),
        "acousticness": round(float(track["acousticness"]), 2),
        "spotify_id": track["track_id"],
    }


def bulk_fill(df: pd.DataFrame, per_genre: int, used_ids: set, used_keys: set, start_id: int) -> list:
    """Most popular single-genre tracks in each genre, skipping anything already in the catalog."""
    genre_counts = df.groupby("track_id")["track_genre"].nunique()
    single = df[df["track_id"].isin(genre_counts[genre_counts == 1].index)]
    single = single[(single["popularity"] >= BULK_MIN_POPULARITY) & single["english_ok"]]
    rows, next_id = [], start_id
    for genre, spotify_genre in GENRE_MAP.items():
        pool = single[single["track_genre"] == spotify_genre].sort_values(
            ["popularity", "track_id"], ascending=[False, True]
        )
        added = 0
        for _, track in pool.iterrows():
            if added >= per_genre:
                break
            key = song_key(track["track_name"], track["artists"])
            if track["track_id"] in used_ids or key in used_keys:
                continue
            used_ids.add(track["track_id"])
            used_keys.add(key)
            rows.append(make_row(next_id, track, genre, infer_mood(track)))
            next_id += 1
            added += 1
        if added < per_genre:
            print(f"  note: only {added} unique songs available for {genre}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_CACHE, help="Path to the Spotify tracks CSV")
    parser.add_argument("--per-genre", type=int, default=DEFAULT_PER_GENRE,
                        help=f"Max bulk-fill songs per genre (default {DEFAULT_PER_GENRE})")
    args = parser.parse_args()

    df = load_dataset(args.source)
    used_ids, used_keys, rows = set(), set(), []
    for i, (genre, mood, pick, target) in enumerate(SLOTS, start=1):
        # Hand-picked songs were checked by hand; auto-picks go through the English filter.
        track = find_song(df, *pick) if pick else auto_pick(df, genre, target, used_ids)
        used_ids.add(track["track_id"])
        used_keys.add(song_key(track["track_name"], track["artists"]))
        rows.append(make_row(i, track, genre, mood))

    rows += bulk_fill(df, args.per_genre, used_ids, used_keys, start_id=len(rows) + 1)

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} songs to {OUTPUT}")


if __name__ == "__main__":
    main()
