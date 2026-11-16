"""
TuneJudge — Streamlit GUI for the Music Recommender Simulation.

Pages (top navigation): AI Assistant (home) · Explore · Insights · Catalog · About.
Explore, Insights and Catalog share the taste profile in the sidebar.

Run from the project root:
    streamlit run src/app.py
"""

import html
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from main import PROFILES  # noqa: E402
from rag import PROVIDER_LABELS, PROVIDERS, TuneJudgeRAG, make_llm  # noqa: E402
from recommender import (  # noqa: E402
    DEFAULT_WEIGHTS,
    load_songs,
    max_score,
    recommend_songs,
    score_breakdown,
    score_song,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES = ["genre", "mood", "energy", "acoustic"]
FEATURE_LABELS = {"genre": "Genre", "mood": "Mood", "energy": "Energy", "acoustic": "Acoustic"}
# Two full color themes. The active one follows `base` in .streamlit/config.toml,
# so switching the whole app (Streamlit widgets + custom cards + charts) is a one-line edit.
# Chart colors were run through the dataviz palette validator against each surface.
THEMES = {
    "light": {
        "features": {"genre": "#2a78d6", "mood": "#eb6834", "energy": "#1baf7a", "acoustic": "#eda100"},
        "bg": "#f7f6f3", "surface": "#ffffff", "surface_2": "#f1f0ec", "track": "#e7e5e0",
        "grid": "#e4e2dc", "border": "rgba(20,20,30,.09)", "border_hover": "rgba(98,80,214,.45)",
        "text": "#1b1b1f", "text_2": "#3f3e3b", "muted": "#6e6d68", "faint": "#8a8984",
        "accent": "#6250d6", "accent_soft": "#5442c4", "accent_glow": "rgba(98,80,214,.16)",
        "green": "#0f8a5f", "green_glow": "rgba(27,175,122,.14)",
        "fit_strong": "#0f7a52", "fit_good": "#5442c4", "fit_weak": "#a05a00",
        "title_grad": "linear-gradient(90deg, #1b1b1f, #6250d6 60%, #0f8a5f)",
        "other_dots": "#c4c3bc", "recommended": "#2a78d6",
        "shadow": "0 1px 2px rgba(20,20,30,.04), 0 6px 20px rgba(20,20,30,.05)",
        "cover": ("72% 60%", "68% 44%"),
    },
    "dark": {
        "features": {"genre": "#4a95ef", "mood": "#e8652f", "energy": "#1cab79", "acoustic": "#c98500"},
        "bg": "#15151c", "surface": "#1d1d25", "surface_2": "#272731", "track": "#2e2e38",
        "grid": "#34343e", "border": "rgba(255,255,255,.11)", "border_hover": "rgba(161,151,245,.45)",
        "text": "#f0efec", "text_2": "#d4d3ca", "muted": "#bebdb5", "faint": "#908f88",
        "accent": "#a197f5", "accent_soft": "#c9c3ff", "accent_glow": "rgba(161,151,245,.5)",
        "green": "#8fe3c4", "green_glow": "rgba(28,171,121,.38)",
        "fit_strong": "#8fe3c4", "fit_good": "#c9c3ff", "fit_weak": "#e6b062",
        "title_grad": "linear-gradient(90deg, #ffffff, #c9c3ff 55%, #8fe3c4)",
        "other_dots": "#6a6973", "recommended": "#4a95ef",
        "shadow": "none",
        "cover": ("78% 62%", "70% 40%"),
    },
}
THEME_NAME = "dark" if st.get_option("theme.base") == "dark" else "light"
T = THEMES[THEME_NAME]

FEATURE_COLORS = T["features"]
SURFACE = T["surface"]
TEXT = T["text"]
TEXT_MUTED = T["muted"]
GRID = T["grid"]

CUSTOM = "Custom"
CURRENT = "Your current profile"

GENRE_STYLE = {  # (emoji, hue) for the generated cover art
    "pop": ("🎤", 330), "indie pop": ("🌇", 300), "lofi": ("🌧️", 220),
    "ambient": ("🪐", 250), "rock": ("🎸", 10), "metal": ("🔥", 0),
    "jazz": ("🎷", 35), "synthwave": ("🌆", 280), "country": ("🤠", 40),
    "hip hop": ("🎧", 45), "r&b": ("💜", 300), "edm": ("⚡", 190),
    "classical": ("🎻", 45), "reggae": ("🌴", 120), "blues": ("🎺", 210),
    "latin": ("💃", 15), "folk": ("🪕", 90),
}

FIT_LEVELS = [  # (min share of max score, label, icon, css class)
    (0.75, "Strong fit", "●", "fit-strong"),
    (0.55, "Good fit", "◐", "fit-good"),
    (0.0, "Weak fit", "○", "fit-weak"),
]

# ---------------------------------------------------------------------------
# Page + styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="TuneJudge", page_icon=":material/graphic_eq:", layout="wide")

CSS_VARS = {
    "bg": T["bg"], "surface": T["surface"], "surface-2": T["surface_2"], "track": T["track"],
    "border": T["border"], "border-hover": T["border_hover"], "text": T["text"], "text-2": T["text_2"],
    "muted": T["muted"], "faint": T["faint"], "accent": T["accent"], "accent-soft": T["accent_soft"],
    "accent-glow": T["accent_glow"], "green": T["green"], "green-glow": T["green_glow"],
    "fit-strong": T["fit_strong"], "fit-good": T["fit_good"], "fit-weak": T["fit_weak"],
    "title-grad": T["title_grad"], "shadow": T["shadow"],
}
st.markdown(
    "<style>:root{" + "".join(f"--tj-{k}:{v};" for k, v in CSS_VARS.items()) + "}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

.block-container { padding-top: 2.5rem; max-width: 1180px; }
.tj, .tj * { font-family: 'Inter', sans-serif; }

.page-head { margin: 2px 0 20px; }
.page-head .eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: 12px; color: var(--tj-accent); font-weight: 600; }
.page-head h1 { font-family: 'Space Grotesk', sans-serif !important; font-size: 34px; line-height: 1.1; margin: 6px 0 6px;
  color: var(--tj-text); padding: 0; }
.page-head p { color: var(--tj-muted); font-size: 15px; margin: 0; max-width: 720px; }
.page-head.hero-ask { text-align: center; margin: 18px auto 22px; }
.page-head.hero-ask h1 { font-size: 46px; background: var(--tj-title-grad); -webkit-background-clip: text;
  background-clip: text; color: transparent; }
.page-head.hero-ask p { margin: 0 auto; }
@media (max-width: 700px){ .page-head.hero-ask h1{font-size:34px} }

.how { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 26px 0 8px; }
.how > div { background: var(--tj-surface); border: 1px solid var(--tj-border); box-shadow: var(--tj-shadow);
  border-radius: 16px; padding: 16px 18px; }
.how .n { font-family: 'Space Grotesk', sans-serif; color: var(--tj-accent); font-weight: 700; font-size: 13px; }
.how b { display: block; color: var(--tj-text); font-size: 15px; margin: 4px 0 4px; }
.how span { color: var(--tj-muted); font-size: 13.5px; line-height: 1.45; }
@media (max-width: 760px){ .how { grid-template-columns: 1fr; } }

.summary { background: var(--tj-surface); border: 1px solid var(--tj-border); border-left: 3px solid var(--tj-accent);
  border-radius: 12px; padding: 12px 16px; color: var(--tj-text-2); font-size: 15px; margin: 4px 0 16px; }
.badge { display: inline-block; font-size: 11.5px; font-weight: 600; padding: 2px 9px; border-radius: 999px;
  border: 1px solid var(--tj-border); color: var(--tj-muted); background: var(--tj-surface-2); }
.badge.ok { color: var(--tj-green); border-color: var(--tj-green); background: transparent; }
.why { margin: 8px 0 0; color: var(--tj-text-2); font-size: 14.5px; line-height: 1.5; }

.steps { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 12px; }
.step { display: inline-flex; flex-direction: column; padding: 7px 12px; border-radius: 12px; font-size: 13px;
  border: 1px solid var(--tj-border); background: var(--tj-surface); color: var(--tj-text); min-width: 120px; }
.step small { color: var(--tj-muted); font-size: 11.5px; }
.step::before { font-weight: 700; margin-right: 4px; }
.step.ok { border-color: var(--tj-green); } .step.ok b::before { content: "✓ "; color: var(--tj-green); }
.step.warn { border-color: var(--tj-fit-weak); } .step.warn b::before { content: "! "; color: var(--tj-fit-weak); }
.step.fail { border-color: #d64545; } .step.fail b::before { content: "✕ "; color: #d64545; }
.step b { font-weight: 600; }

.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 14px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
  background: var(--tj-surface); border: 1px solid var(--tj-border); color: var(--tj-text-2); font-size: 13px; }
.chip b { color: var(--tj-text); font-weight: 600; }

.stat { background: var(--tj-surface); border: 1px solid var(--tj-border); box-shadow: var(--tj-shadow); border-radius: 16px; padding: 14px 18px; }
.stat .label { color: var(--tj-muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }
.stat .value { font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; color: var(--tj-text); line-height: 1.2; }
.stat .sub { color: var(--tj-muted); font-size: 12.5px; }

.legend { display: flex; flex-wrap: wrap; gap: 16px; color: var(--tj-text-2); font-size: 13px; margin: 4px 0 12px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }

.card { display: grid; grid-template-columns: 34px 92px 1fr 92px; gap: 18px; align-items: center;
  background: var(--tj-surface); border: 1px solid var(--tj-border); box-shadow: var(--tj-shadow);
  border-radius: 18px; padding: 16px 18px; margin-bottom: 12px;
  transition: transform .15s ease, border-color .15s ease; }
.card:hover { transform: translateY(-2px); border-color: var(--tj-border-hover); }
.rank { font-family: 'Space Grotesk', sans-serif; font-size: 26px; font-weight: 700; color: var(--tj-faint); text-align: center; }
.card.top .rank { color: var(--tj-accent); }
.cover { width: 92px; height: 92px; border-radius: 14px; display: grid; place-items: center; font-size: 38px;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.18); }
.meta .title { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: var(--tj-text); }
.meta .artist { color: var(--tj-muted); font-size: 14px; margin-bottom: 8px; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; margin-right: 6px;
  background: var(--tj-surface-2); color: var(--tj-text-2); border: 1px solid var(--tj-border); }
.pill.hit { border-color: var(--tj-accent); color: var(--tj-text); }
.facts { color: var(--tj-faint); font-size: 12px; margin-left: 4px; }
.segbar { display: flex; gap: 2px; height: 8px; border-radius: 4px; background: var(--tj-track); overflow: hidden; margin: 10px 0 8px; }
.segbar span { height: 100%; }
.reasons { display: flex; flex-wrap: wrap; gap: 6px; }
.reason { font-size: 12px; color: var(--tj-text-2); background: var(--tj-surface-2); border-radius: 8px; padding: 3px 8px; }
.reason i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
.ring { width: 84px; height: 84px; border-radius: 50%; display: grid; place-items: center; margin-left: auto; }
.ring .inner { width: 68px; height: 68px; border-radius: 50%; background: var(--tj-surface); display: grid; place-items: center; text-align: center; }
.ring .pct { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 20px; color: var(--tj-text); line-height: 1; }
.ring .of { font-size: 10.5px; color: var(--tj-muted); margin-top: 2px; }
.fit { font-size: 11.5px; text-align: center; margin-top: 6px; font-weight: 500; }
.fit-strong { color: var(--tj-fit-strong); } .fit-good { color: var(--tj-fit-good); } .fit-weak { color: var(--tj-fit-weak); }
@media (max-width: 760px){
  .card { grid-template-columns: 64px 1fr; }
  .rank { display: none; } .cover { width: 64px; height: 64px; font-size: 28px; }
  .scorecol { grid-column: 1 / -1; display: flex; align-items: center; gap: 12px; }
  .ring { margin-left: 0; }
}

.mini { display: flex; align-items: center; gap: 12px; padding: 10px 12px; border-radius: 12px; background: var(--tj-surface);
  border: 1px solid var(--tj-border); box-shadow: var(--tj-shadow); margin-bottom: 8px; }
.mini.shared { border-color: var(--tj-green); }
.mini .c { width: 42px; height: 42px; border-radius: 10px; display: grid; place-items: center; font-size: 20px; flex: none; }
.mini .t { color: var(--tj-text); font-weight: 600; font-size: 14px; }
.mini .a { color: var(--tj-muted); font-size: 12px; }
.mini .s { margin-left: auto; color: var(--tj-text); font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
.mini .tag { font-size: 11px; color: var(--tj-green); margin-left: 6px; font-weight: 600; }

.player { display: block; width: 100%; height: 80px; border: 0; border-radius: 12px; margin-top: 12px; }
.mini .t a { color: var(--tj-faint); text-decoration: none; margin-left: 6px; font-size: 12px; }
.mini .t a:hover { color: var(--tj-accent); }
.footer { color: var(--tj-faint); font-size: 12.5px; text-align: center; margin: 36px 0 10px; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data + helpers
# ---------------------------------------------------------------------------

@st.cache_data
def get_songs():
    return load_songs(str(ROOT / "data" / "songs.csv"))


@st.cache_resource
def get_rag(provider: str, auto_switch: bool = True):
    """One RAG pipeline (indexes + model clients) per provider, shared across reruns.

    With auto_switch, the other providers (those with a key) take over when this one is busy.
    """
    fallbacks = [make_llm(p) for p in PROVIDERS if p != provider] if auto_switch else []
    return TuneJudgeRAG(get_songs(), llm=make_llm(provider), fallbacks=fallbacks)


def esc(text) -> str:
    return html.escape(str(text))


def cover_style(genre: str) -> tuple:
    emoji, hue = GENRE_STYLE.get(genre.strip().lower(), ("🎵", 260))
    light, deep = T["cover"]
    bg = f"linear-gradient(135deg, hsl({hue} {light}), hsl({(hue + 50) % 360} {deep}))"
    return emoji, bg


def spotify_id_of(song) -> str:
    """Spotify track id for a song dict/row, or '' if the catalog has none."""
    value = song.get("spotify_id", "") if isinstance(song, dict) else getattr(song, "spotify_id", "")
    return value if isinstance(value, str) else ""


def spotify_player(track_id: str) -> str:
    if not track_id:
        return ""
    return (
        f'<iframe class="player" src="https://open.spotify.com/embed/track/{esc(track_id)}?utm_source=generator" '
        'height="80" frameborder="0" loading="lazy" '
        'allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>'
    )


def fit_level(share: float) -> tuple:
    for threshold, label, icon, css in FIT_LEVELS:
        if share >= threshold:
            return label, icon, css
    return FIT_LEVELS[-1][1:]


def score_table(prefs: dict, songs: list, weights: dict) -> pd.DataFrame:
    """Score every song; one row per song with per-feature points, sorted like recommend_songs."""
    rows = []
    for song in songs:
        parts, reasons = score_breakdown(prefs, song, weights)
        score, _ = score_song(prefs, song, weights)
        rows.append({**song, **{f"pts_{f}": parts[f] for f in FEATURES}, "score": score, "reasons": reasons})
    df = pd.DataFrame(rows)
    # Stable sort keeps catalog order for ties, matching Python's sorted() in recommend_songs.
    return df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


def chart_config(chart):
    return (
        chart.configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=TEXT_MUTED, titleColor=TEXT_MUTED, gridColor=GRID, domainColor=GRID,
                        tickColor=GRID, labelFont="Inter", titleFont="Inter", labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor=TEXT, titleColor=TEXT_MUTED, labelFont="Inter", titleFont="Inter",
                          orient="top", labelFontSize=12)
    )


# ---------------------------------------------------------------------------
# Session state + sidebar controls
# ---------------------------------------------------------------------------

def apply_preset():
    name = st.session_state.preset
    if name not in PROFILES:
        return
    p = PROFILES[name]
    raw = float(p["target_energy"])
    clamped = min(max(raw, 0.0), 1.0)
    st.session_state.genre = p["favorite_genre"]
    st.session_state.mood = p["favorite_mood"]
    st.session_state.energy = clamped
    st.session_state.acoustic = p["likes_acoustic"]
    st.session_state.clamped_from = raw if raw != clamped else None


def mark_custom():
    st.session_state.preset = CUSTOM
    st.session_state.clamped_from = None


def reset_weights():
    for f in FEATURES:
        st.session_state[f"w_{f}"] = DEFAULT_WEIGHTS[f]


if "preset" not in st.session_state:
    st.session_state.preset = "Chill Lofi"
    apply_preset()
    reset_weights()
    st.session_state.k = 5
    default_provider = os.environ.get("LLM_PROVIDER", "claude").strip().lower()
    st.session_state.ai_provider = default_provider if default_provider in PROVIDERS else "claude"
    st.session_state.ai_auto = True
    st.session_state.ai_k = 5

# Streamlit forgets a widget's value when the page showing it isn't rendered.
# Re-assigning the keys each run keeps the profile and settings while you switch pages.
PERSIST = ["preset", "genre", "mood", "energy", "acoustic", "k", "ai_provider", "ai_auto", "ai_k"] + \
          [f"w_{f}" for f in FEATURES]
for _key in PERSIST:
    if _key in st.session_state:
        st.session_state[_key] = st.session_state[_key]

songs = get_songs()
catalog_genres = sorted({s["genre"] for s in songs})
catalog_moods = sorted({s["mood"] for s in songs})


def options_with(base: list, extra: list, current: str) -> list:
    out = list(base)
    for value in extra + [current]:
        if value not in out:
            out.append(value)
    return out

def page_header(eyebrow: str, title: str, text: str, hero: bool = False) -> None:
    st.markdown(
        f'<div class="tj page-head {"hero-ask" if hero else ""}"><div class="eyebrow">{esc(eyebrow)}</div>'
        f"<h1>{esc(title)}</h1><p>{esc(text)}</p></div>",
        unsafe_allow_html=True,
    )


def reason_feature(reason: str) -> str:
    if reason.startswith(("genre", "related genre")):
        return "genre"
    if reason.startswith("mood"):
        return "mood"
    if reason.startswith("energy"):
        return "energy"
    return "acoustic"


def chips_html(profile: dict, lead: str = "", extra: str = "") -> str:
    lead_html = f'<span class="chip">{esc(lead)}</span>' if lead else ""
    return (
        f'<div class="tj chips">{lead_html}'
        f'<span class="chip">Genre <b>{esc(str(profile["favorite_genre"]).strip())}</b></span>'
        f'<span class="chip">Mood <b>{esc(str(profile["favorite_mood"]).strip())}</b></span>'
        f'<span class="chip">Energy <b>{profile["target_energy"]:.2f}</b></span>'
        f'<span class="chip"><b>{"Acoustic" if profile["likes_acoustic"] else "Non-acoustic"}</b></span>'
        f"{extra}</div>"
    )


def score_ring(share: float, score: float, max_pts: float) -> str:
    label, icon, css = fit_level(share)
    ring = f"conic-gradient({T['accent']} 0 {share * 360:.1f}deg, {T['track']} {share * 360:.1f}deg 360deg)"
    return (
        f'<div class="scorecol"><div class="ring" style="background:{ring}"><div class="inner"><div>'
        f'<div class="pct">{share:.0%}</div><div class="of">{score:.2f} / {max_pts:.1f}</div></div></div></div>'
        f'<div class="fit {css}">{icon} {label}</div></div>'
    )


# ---------------------------------------------------------------------------
# Shared taste profile (sidebar on Explore / Insights / Catalog)
# ---------------------------------------------------------------------------

def profile_sidebar() -> dict:
    """Render the taste-profile sidebar and return everything the manual pages need."""
    with st.sidebar:
        st.markdown("### Taste profile")
        st.caption("Shared by Explore, Insights and Catalog.")
        st.selectbox("Start from a preset", [CUSTOM] + list(PROFILES.keys()), key="preset", on_change=apply_preset,
                     help="Includes the adversarial edge-case profiles from main.py.")
        st.selectbox("Favorite genre",
                     options_with(catalog_genres, [p["favorite_genre"] for p in PROFILES.values()], st.session_state.genre),
                     key="genre", on_change=mark_custom, accept_new_options=True,
                     help="Pick one or type your own. Matching ignores case and extra spaces.")
        st.selectbox("Favorite mood",
                     options_with(catalog_moods, [p["favorite_mood"] for p in PROFILES.values()], st.session_state.mood),
                     key="mood", on_change=mark_custom, accept_new_options=True)
        st.slider("Target energy", 0.0, 1.0, step=0.01, key="energy", on_change=mark_custom,
                  help="0 = calm, 1 = intense. Songs closest to this value score highest.")
        st.toggle("I like acoustic sound", key="acoustic", on_change=mark_custom)
        st.slider("Number of songs", 3, 10, key="k")

        with st.expander("Scoring lab: tune the weights", icon=":material/science:"):
            st.caption("Change how much each rule is worth and watch the ranking shift. "
                       "A related genre earns half the genre weight.")
            for f in FEATURES:
                st.slider(FEATURE_LABELS[f], 0.0, 4.0, step=0.25, key=f"w_{f}")
            st.button("Reset to defaults", on_click=reset_weights, width="stretch")

    weights = {f: st.session_state[f"w_{f}"] for f in FEATURES}
    max_pts = max_score(weights)
    prefs = {
        "favorite_genre": st.session_state.genre,
        "favorite_mood": st.session_state.mood,
        "target_energy": st.session_state.energy,
        "likes_acoustic": st.session_state.acoustic,
    }
    k = st.session_state.k

    st.markdown(chips_html(prefs, extra=f'<span class="chip">Max score <b>{max_pts:.2f}</b></span>'),
                unsafe_allow_html=True)
    if max_pts == 0:
        st.error("All weights are zero, so every song scores 0. Raise at least one weight in the Scoring lab.")
        st.stop()
    if st.session_state.get("clamped_from") is not None:
        st.info(
            f"This preset asks for energy **{st.session_state.clamped_from}**, which is outside the 0–1 scale. "
            f"TuneJudge clamped it to **{prefs['target_energy']:.2f}** so energy points can't go negative."
        )

    table = score_table(prefs, songs, weights)
    top = table.head(k)
    genre_norm = prefs["favorite_genre"].strip().lower()
    mood_norm = prefs["favorite_mood"].strip().lower()
    return {
        "prefs": prefs, "weights": weights, "max": max_pts, "k": k, "table": table, "top": top,
        "best_share": top["score"].iloc[0] / max_pts,
        "genre_count": sum(s["genre"].lower() == genre_norm for s in songs),
        "mood_count": sum(s["mood"].lower() == mood_norm for s in songs),
    }


def match_warnings(ctx: dict) -> None:
    missing = []
    if ctx["genre_count"] == 0:
        missing.append(f"genre **{ctx['prefs']['favorite_genre'].strip()}**")
    if ctx["mood_count"] == 0:
        missing.append(f"mood **{ctx['prefs']['favorite_mood'].strip()}**")
    if missing:
        st.warning(
            f"No songs in the catalog have your {' or '.join(missing)}. These picks are ranked mostly on energy "
            "and acoustic feel, so treat them as loose suggestions rather than real matches."
        )
    elif ctx["best_share"] < 0.55:
        st.warning("No strong match in this catalog. Even the top pick earns less than 55% of the possible score.")


# ---------------------------------------------------------------------------
# Page: AI Assistant (home)
# ---------------------------------------------------------------------------

EXAMPLES = {
    "Deep focus": "calm music for studying",
    "Road trip": "upbeat songs for a road trip",
    "Gym session": "heavy stuff for lifting weights at the gym",
    "Rainy day": "cozy rainy day with tea",
    "Date night": "romantic dinner date night",
    "Heartbreak": "i just got dumped and want to wallow",
    "Falling asleep": "songs to fall asleep to",
}


def use_example():
    choice = st.session_state.get("example")
    if choice:
        st.session_state.ask_text = EXAMPLES[choice]
        st.session_state.ask_run = True
    st.session_state.example = None


def pipeline_steps(res) -> list:
    """(label, state, detail) for each pipeline stage, read from the result + trace."""
    trace = res.trace
    if any(t.startswith("input guardrail: blocked") for t in trace):
        return [("Input check", "fail", "blocked before any AI call")]
    steps = [("Input check", "ok", "passed"), ("Knowledge retrieved", "ok", f"{len(res.notes)} notes")]
    who = PROVIDER_LABELS.get(res.provider, res.provider)
    if res.status == "error":
        return steps + [("AI reads request", "fail", "model unavailable")]
    if res.status == "blocked":
        return steps + [("AI reads request", "fail", f"{who} declined")]
    if res.status == "off_topic":
        return steps + [("AI reads request", "warn", "not a music request")]
    steps.append(("AI reads request", "ok", who + (" (auto-switched)" if res.fallback_from else "")))
    steps.append(("Profile validated", "warn" if res.fixes else "ok",
                  f"{len(res.fixes)} fix{'es' if len(res.fixes) != 1 else ''}" if res.fixes else "no fixes needed"))
    retr = next((t.split(":", 1)[1].strip() for t in trace if t.startswith("retriever")), "")
    steps.append(("Songs retrieved", "ok", retr.replace(" candidate songs", " candidates")[:40]))
    steps.append(("Ranked by scorer", "ok", f"top {len(res.recommendations)}"))
    steps.append(("Explanation verified", "ok" if res.ai_explanation else "warn",
                  "grounded in data" if res.ai_explanation else "used scorer's reasons"))
    return steps


def page_assistant():
    head_col, settings_col = st.columns([6, 1], vertical_alignment="bottom")
    with settings_col:
        with st.popover("Settings", icon=":material/tune:", width="stretch"):
            st.radio("AI model", sorted(PROVIDERS), key="ai_provider",
                     format_func=lambda p: {"claude": "Claude (Anthropic)", "gemini": "Gemini (Google)",
                                            "openai": "OpenAI (GPT)"}.get(p, p),
                     help="Default comes from LLM_PROVIDER in .env. Retrieval, scoring and guardrails are the same for all.")
            st.toggle("Auto-switch if this model is busy", key="ai_auto",
                      help="If the chosen model times out, is overloaded or rate-limited, "
                           "try the other models that have a key in .env.")
            st.slider("Number of songs", 3, 10, key="ai_k")
    provider, auto_switch, k = st.session_state.ai_provider, st.session_state.ai_auto, st.session_state.ai_k
    rag = get_rag(provider, auto_switch)
    llm = rag.get_llm()
    with head_col:
        page_header("AI music assistant", "Find music for any moment",
                    "Describe a mood, an activity or a moment. The assistant reads your request, searches "
                    f"{len(songs):,} real tracks, and checks every explanation against the data.", hero=True)

    with st.form("ask_form", clear_on_submit=False, border=False):
        q_col, b_col = st.columns([6, 1], vertical_alignment="bottom")
        query = q_col.text_input("Your request", key="ask_text", max_chars=500, label_visibility="collapsed",
                                 placeholder="e.g. calm music for studying on a rainy night")
        asked = b_col.form_submit_button("Search", type="primary", icon=":material/search:", width="stretch")
    st.pills("Try an example", list(EXAMPLES), key="example", on_change=use_example, label_visibility="collapsed")

    who = PROVIDER_LABELS.get(provider, provider)
    if not llm.has_key():
        st.info(f"{who} needs an API key. Paste it into `{llm.key_names[0]}=` in the project's `.env` file, "
                "then restart the app. You can also pick another model under Settings.", icon=":material/key:")

    if asked or st.session_state.pop("ask_run", False):
        with st.spinner("Finding songs: reading your request → retrieving → ranking → checking the explanation"):
            st.session_state.ask_result = rag.run(st.session_state.ask_text, k=k,
                                                  weights={f: st.session_state[f"w_{f}"] for f in FEATURES})

    res = st.session_state.get("ask_result")
    if res is None:
        st.markdown(
            '<div class="tj how">'
            '<div><div class="n">01</div><b>Understands your request</b><span>An AI model turns plain words into a '
            "genre, mood and energy, guided by notes retrieved from a music knowledge base.</span></div>"
            '<div><div class="n">02</div><b>Ranks with a transparent scorer</b><span>The same rule-based scorer used '
            "everywhere in TuneJudge ranks the songs. You can inspect it on the Insights page.</span></div>"
            '<div><div class="n">03</div><b>Checks every explanation</b><span>Guardrails block unsafe input, fix '
            "out-of-range values, and reject explanations that mention anything not in the data.</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model: {who} · `{llm.model}`" + (" · auto-switch on" if auto_switch else ""))
        return

    max_pts = max_score({f: st.session_state[f"w_{f}"] for f in FEATURES})
    if res.fallback_from:
        st.info(f"{PROVIDER_LABELS.get(res.fallback_from, res.fallback_from)} was busy, so "
                f"{PROVIDER_LABELS.get(res.provider, res.provider)} answered this one.", icon=":material/sync_alt:")
    if res.status != "ok":
        (st.error if res.status == "error" else st.warning)(res.message)
    else:
        st.markdown(chips_html(res.profile, lead="Understood as"), unsafe_allow_html=True)
        if res.summary:
            st.markdown(f'<div class="tj summary">{esc(res.summary)}</div>', unsafe_allow_html=True)
        if res.fixes:
            st.info("Guardrail adjusted the AI's reading: " + "; ".join(res.fixes), icon=":material/shield:")
        if not res.ai_explanation:
            st.warning("The AI explanation didn't pass the grounding check, so you're seeing the scorer's own reasons.",
                       icon=":material/shield:")
        best = res.recommendations[0]["score"] / max_pts if res.recommendations and max_pts else 0
        if best < 0.55:
            st.warning("No strong match in this catalog. Even the top pick earns less than 55% of the possible score.")

        badge = ('<span class="badge ok">AI explanation · verified</span>' if res.ai_explanation
                 else '<span class="badge">Scorer\'s reasons</span>')
        for rank, rec in enumerate(res.recommendations, start=1):
            song = rec["song"]
            share = rec["score"] / max_pts if max_pts else 0
            emoji, bg = cover_style(song["genre"])
            st.markdown(
                f'<div class="tj card {"top" if rank == 1 else ""}">'
                f'<div class="rank">{rank}</div>'
                f'<div class="cover" style="background:{bg}">{emoji}</div>'
                f'<div class="meta"><div class="title">{esc(song["title"])}</div><div class="artist">{esc(song["artist"])}</div>'
                f'<span class="pill">{esc(song["genre"])}</span><span class="pill">{esc(song["mood"])}</span>'
                f'<span class="facts">energy {song["energy"]:.2f} · acoustic {song["acousticness"]:.2f}</span>'
                f'<p class="why">{esc(rec["why"])}</p><div style="margin-top:8px">{badge}</div>'
                f'{spotify_player(spotify_id_of(song))}</div>'
                f'{score_ring(share, rec["score"], max_pts)}'
                "</div>",
                unsafe_allow_html=True,
            )

    with st.expander("How this answer was built", icon=":material/account_tree:"):
        st.markdown(
            '<div class="tj steps">' + "".join(
                f'<div class="step {state}"><b>{esc(label)}</b><small>{esc(detail)}</small></div>'
                for label, state, detail in pipeline_steps(res)
            ) + "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{PROVIDER_LABELS.get(res.provider, res.provider)} · {res.model} · {res.latency_s:.2f}s · "
                   f"retrieved notes: {', '.join(res.notes) or 'none'}")
        if res.grounding_problems:
            st.markdown("**Grounding problems found:** " + "; ".join(res.grounding_problems))
        st.code("\n".join(res.trace) or "(no steps)", language=None)


# ---------------------------------------------------------------------------
# Page: Explore (manual profile → ranked picks)
# ---------------------------------------------------------------------------

def page_explore():
    page_header("Explore", "Recommendations from your taste profile",
                "Set a genre, mood and energy in the sidebar. Every song in the catalog is scored against it, "
                "and each pick shows exactly where its points came from.")
    ctx = profile_sidebar()
    match_warnings(ctx)
    top, max_pts, k = ctx["top"], ctx["max"], ctx["k"]

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, "Best match", f"{ctx['best_share']:.0%}", f"{top['title'].iloc[0]}"),
        (c2, f"Avg fit · top {k}", f"{(top['score'].mean() / max_pts):.0%}", "share of max score"),
        (c3, "Songs in your genre", f"{ctx['genre_count']}", f"of {len(songs):,} in catalog"),
        (c4, "Songs in your mood", f"{ctx['mood_count']}", f"of {len(songs):,} in catalog"),
    ]
    for col, label, value, sub in stats:
        col.markdown(
            f'<div class="tj stat"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    st.markdown('<div class="tj legend">' + "".join(
        f'<span><i style="background:{FEATURE_COLORS[f]}"></i>{FEATURE_LABELS[f]}</span>' for f in FEATURES
    ) + "</div>", unsafe_allow_html=True)
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        share = row.score / max_pts
        emoji, bg = cover_style(row.genre)
        segs = "".join(
            f'<span title="{FEATURE_LABELS[f]}: +{getattr(row, "pts_" + f):.2f}" '
            f'style="width:{getattr(row, "pts_" + f) / max_pts * 100:.2f}%;background:{FEATURE_COLORS[f]}"></span>'
            for f in FEATURES if getattr(row, "pts_" + f) > 0
        )
        reasons = "".join(
            f'<span class="reason"><i style="background:{FEATURE_COLORS[reason_feature(r)]}"></i>{esc(r)}</span>'
            for r in row.reasons
        )
        genre_hit = "hit" if row.pts_genre > 0 else ""
        mood_hit = "hit" if row.pts_mood > 0 else ""
        st.markdown(
            f'<div class="tj card {"top" if rank == 1 else ""}">'
            f'<div class="rank">{rank}</div>'
            f'<div class="cover" style="background:{bg}">{emoji}</div>'
            f'<div class="meta"><div class="title">{esc(row.title)}</div><div class="artist">{esc(row.artist)}</div>'
            f'<span class="pill {genre_hit}">{esc(row.genre)}</span><span class="pill {mood_hit}">{esc(row.mood)}</span>'
            f'<span class="facts">{row.tempo_bpm:.0f} BPM · energy {row.energy:.2f} · acoustic {row.acousticness:.2f}</span>'
            f'<div class="segbar">{segs}</div><div class="reasons">{reasons}</div>'
            f'{spotify_player(spotify_id_of(row))}</div>'
            f'{score_ring(share, row.score, max_pts)}'
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Page: Insights (score breakdown + profile comparison)
# ---------------------------------------------------------------------------

def page_insights():
    page_header("Insights", "How the scorer makes its decisions",
                "See where each recommendation's points come from, and compare how two listeners get served. "
                "Tune the rule weights in the sidebar's Scoring lab.")
    ctx = profile_sidebar()
    top, max_pts, k, prefs, weights = ctx["top"], ctx["max"], ctx["k"], ctx["prefs"], ctx["weights"]
    tab_breakdown, tab_compare = st.tabs(["Score breakdown", "Compare profiles"])

    with tab_breakdown:
        st.caption("Each bar is one recommendation, split by scoring rule. The bar's full width is the best possible score.")
        long = top.melt(id_vars=["title", "artist", "score"], value_vars=[f"pts_{f}" for f in FEATURES],
                        var_name="feature", value_name="points")
        long["feature"] = long["feature"].str.replace("pts_", "").map(FEATURE_LABELS)
        long["order"] = long["feature"].map({FEATURE_LABELS[f]: i for i, f in enumerate(FEATURES)})
        breakdown = (
            alt.Chart(long)
            .mark_bar(height=22, stroke=SURFACE, strokeWidth=2)
            .encode(
                x=alt.X("points:Q", stack="zero", title="Points", scale=alt.Scale(domain=[0, max_pts]),
                        axis=alt.Axis(tickCount=6)),
                y=alt.Y("title:N", sort=list(top["title"]), title=None, axis=alt.Axis(labelLimit=220)),
                color=alt.Color("feature:N", title=None,
                                scale=alt.Scale(domain=[FEATURE_LABELS[f] for f in FEATURES],
                                                range=[FEATURE_COLORS[f] for f in FEATURES])),
                order=alt.Order("order:Q"),
                tooltip=[
                    alt.Tooltip("title:N", title="Song"),
                    alt.Tooltip("feature:N", title="Rule"),
                    alt.Tooltip("points:Q", title="Points", format=".2f"),
                    alt.Tooltip("score:Q", title="Total", format=".2f"),
                ],
            )
            .properties(height=max(200, 46 * len(top)))
        )
        totals = (
            alt.Chart(top)
            .mark_text(align="left", dx=6, color=TEXT, font="Inter", fontSize=12)
            .encode(x="score:Q", y=alt.Y("title:N", sort=list(top["title"])), text=alt.Text("score:Q", format=".2f"))
        )
        st.altair_chart(chart_config(breakdown + totals), width="stretch", theme=None)

        with st.expander("Show as a table"):
            st.dataframe(
                top[["title", "artist", "pts_genre", "pts_mood", "pts_energy", "pts_acoustic", "score"]].rename(
                    columns={"title": "Song", "artist": "Artist", "pts_genre": "Genre", "pts_mood": "Mood",
                             "pts_energy": "Energy", "pts_acoustic": "Acoustic", "score": "Total"}
                ),
                hide_index=True, width="stretch",
                column_config={c: st.column_config.NumberColumn(format="%.2f")
                               for c in ["Genre", "Mood", "Energy", "Acoustic", "Total"]},
            )

        st.markdown("##### What this tells you")
        energy_share = top["pts_energy"].sum() / top["score"].sum()
        matched = int(((top["pts_genre"] > 0) | (top["pts_mood"] > 0)).sum())
        gc = ctx["genre_count"]
        st.markdown(
            f"- **Energy supplies {energy_share:.0%}** of the points across your top {k}.\n"
            f"- **{matched} of {k}** picks match your genre or mood; the rest got in on energy and acoustic feel alone.\n"
            f"- Your genre appears in **{gc}** catalog song{'s' if gc != 1 else ''}, "
            f"so at most {min(gc, k)} pick{'s' if min(gc, k) != 1 else ''} can earn full genre points."
        )

    with tab_compare:
        st.caption("Pick two profiles. Songs that appear in both lists are outlined in green.")
        choices = [CURRENT] + list(PROFILES.keys())
        left_col, right_col = st.columns(2)
        left_name = left_col.selectbox("Profile A", choices, index=0, key="cmp_a")
        right_name = right_col.selectbox("Profile B", choices, index=choices.index("Deep Intense Rock"), key="cmp_b")

        def prefs_for(name):
            if name == CURRENT:
                return prefs
            p = dict(PROFILES[name])
            p["target_energy"] = min(max(float(p["target_energy"]), 0.0), 1.0)
            return p

        recs_a = recommend_songs(prefs_for(left_name), songs, k=k, weights=weights)
        recs_b = recommend_songs(prefs_for(right_name), songs, k=k, weights=weights)
        shared = {s["id"] for s, _, _ in recs_a} & {s["id"] for s, _, _ in recs_b}

        for col, recs in ((left_col, recs_a), (right_col, recs_b)):
            items = []
            for song, score, _ in recs:
                emoji, bg = cover_style(song["genre"])
                is_shared = song["id"] in shared
                sid = spotify_id_of(song)
                link = (f'<a href="https://open.spotify.com/track/{esc(sid)}" target="_blank" '
                        f'title="Open in Spotify">↗</a>') if sid else ""
                items.append(
                    f'<div class="tj mini {"shared" if is_shared else ""}"><div class="c" style="background:{bg}">{emoji}</div>'
                    f'<div><div class="t">{esc(song["title"])}{"<span class=tag>✓ in both</span>" if is_shared else ""}'
                    f'{link}</div>'
                    f'<div class="a">{esc(song["artist"])} · {esc(song["genre"])} / {esc(song["mood"])}</div></div>'
                    f'<div class="s">{score / max_pts:.0%}</div></div>'
                )
            col.markdown("".join(items), unsafe_allow_html=True)

        st.metric("Shared recommendations", f"{len(shared)} of {k}",
                  help="High overlap between very different profiles means energy is doing most of the work.")


# ---------------------------------------------------------------------------
# Page: Catalog
# ---------------------------------------------------------------------------

def page_catalog():
    page_header("Catalog", f"All {len(songs):,} songs through your profile",
                "Every track placed by energy and acoustic feel, plus the full ranking for the profile in the sidebar.")
    ctx = profile_sidebar()
    table, max_pts, k, prefs = ctx["table"], ctx["max"], ctx["k"], ctx["prefs"]

    st.caption("Blue dots are your picks, labeled by rank; the dashed line is your target energy. Hover any dot for details.")
    plot = table.copy()
    plot["group"] = ["Recommended" if i < k else "Other songs" for i in range(len(plot))]
    plot["match"] = plot["score"] / max_pts
    plot["rank_label"] = [f"#{i + 1}" for i in range(len(plot))]
    plot["spotify_url"] = [f"https://open.spotify.com/track/{t}" if isinstance(t, str) and t else None
                           for t in plot.get("spotify_id", pd.Series([None] * len(plot)))]
    base = alt.Chart(plot.iloc[::-1])  # draw recommended points last so they sit on top
    points = base.mark_circle(stroke=SURFACE, strokeWidth=1).encode(
        x=alt.X("energy:Q", title="Energy", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=10)),
        y=alt.Y("acousticness:Q", title="Acousticness", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("group:N", title=None,
                        scale=alt.Scale(domain=["Recommended", "Other songs"], range=[T["recommended"], T["other_dots"]])),
        # With 900+ songs, background dots stay small and translucent so your picks stand out.
        size=alt.Size("group:N", scale=alt.Scale(domain=["Recommended", "Other songs"], range=[200, 34]), legend=None),
        opacity=alt.Opacity("group:N", scale=alt.Scale(domain=["Recommended", "Other songs"], range=[1.0, 0.45]), legend=None),
        tooltip=[
            alt.Tooltip("title:N", title="Song"),
            alt.Tooltip("artist:N", title="Artist"),
            alt.Tooltip("genre:N", title="Genre"),
            alt.Tooltip("mood:N", title="Mood"),
            alt.Tooltip("score:Q", title="Score", format=".2f"),
            alt.Tooltip("match:Q", title="Match", format=".0%"),
        ],
    )
    labels = base.transform_filter(alt.datum.group == "Recommended").mark_text(
        align="left", dx=9, dy=-9, color=TEXT, font="Inter", fontSize=12, fontWeight=600
    ).encode(x="energy:Q", y="acousticness:Q", text="rank_label:N")
    target = alt.Chart(pd.DataFrame({"energy": [prefs["target_energy"]]})).mark_rule(
        color=T["accent"], strokeDash=[5, 4], strokeWidth=2
    ).encode(x="energy:Q")
    st.altair_chart(chart_config((target + points + labels).properties(height=440)), width="stretch", theme=None)

    st.markdown("##### Full ranking")
    plot["rank"] = range(1, len(plot) + 1)
    search_col, genre_col = st.columns([2, 1])
    query = search_col.text_input("Search", placeholder="Song or artist", label_visibility="collapsed")
    genre_filter = genre_col.multiselect("Genres", catalog_genres, placeholder="All genres", label_visibility="collapsed")
    shown = plot
    if query:
        q = query.strip().lower()
        shown = shown[shown["title"].str.lower().str.contains(q, regex=False)
                      | shown["artist"].str.lower().str.contains(q, regex=False)]
    if genre_filter:
        shown = shown[shown["genre"].isin(genre_filter)]
    st.caption(f"Showing {len(shown):,} of {len(plot):,} songs, ranked for your profile.")
    st.dataframe(
        shown[["rank", "title", "artist", "genre", "mood", "energy", "acousticness", "tempo_bpm", "score", "match",
               "spotify_url"]].rename(
            columns={"rank": "#", "title": "Song", "artist": "Artist", "genre": "Genre", "mood": "Mood",
                     "energy": "Energy", "acousticness": "Acoustic", "tempo_bpm": "BPM", "score": "Score",
                     "match": "Match", "spotify_url": "Listen"}
        ),
        hide_index=True, width="stretch", height=420,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "BPM": st.column_config.NumberColumn(format="%d"),
            "Match": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
            "Listen": st.column_config.LinkColumn(display_text="Open in Spotify ↗"),
        },
    )


# ---------------------------------------------------------------------------
# Page: About (architecture, guardrails, evaluation)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def offline_eval() -> dict:
    """Retrieval + guardrail metrics from scripts/eval_rag.py (offline part only: no API calls)."""
    import contextlib
    import io

    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_rag

    with contextlib.redirect_stdout(io.StringIO()):
        return eval_rag.offline(TuneJudgeRAG(get_songs(), llm=object(), log_path=None), get_songs())


def page_about():
    page_header("About", "How TuneJudge works",
                "A content-based recommender with an AI layer on top: retrieval-augmented generation, "
                "three guardrails, automated tests and an evaluation harness.")

    st.markdown("##### Pipeline")
    st.markdown(
        "1. **Input guardrail.** Empty, overlong or prompt-injection requests are refused before any AI call.\n"
        "2. **Retrieve knowledge.** A TF-IDF index finds the most relevant genre, mood and activity notes.\n"
        "3. **AI reads the request.** Claude, Gemini or OpenAI returns a structured taste profile.\n"
        "4. **Profile guardrail.** Genre and mood are snapped to real catalog values and energy is clamped to 0–1.\n"
        "5. **Retrieve and rank.** Candidate songs are retrieved, then ranked by the transparent rule-based scorer.\n"
        "6. **AI explains the picks**, using only the retrieved songs and notes.\n"
        "7. **Grounding guardrail.** The explanation is rejected if it names other songs or invents numbers; "
        "the scorer's own reasons are shown instead."
    )

    st.markdown("##### Reliability")
    c1, c2, c3 = st.columns(3)
    for col, label, value, sub in (
        (c1, "Timeout per AI call", f"{os.environ.get('LLM_TIMEOUT_S', '30')}s", "set with LLM_TIMEOUT_S"),
        (c2, "AI providers", f"{len(PROVIDERS)}", " · ".join(PROVIDER_LABELS[p] for p in sorted(PROVIDERS))),
        (c3, "Auto-switch", "On" if st.session_state.ai_auto else "Off", "when a provider is busy"),
    ):
        col.markdown(f'<div class="tj stat"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
                     f'<div class="sub">{esc(sub)}</div></div>', unsafe_allow_html=True)

    st.markdown("##### Offline evaluation")
    st.caption("Computed now from scripts/eval_rag.py on the golden query set. No API calls.")
    try:
        ev = offline_eval()
        c1, c2, c3 = st.columns(3)
        for col, label, value, sub in (
            (c1, "Knowledge hit@4", f"{ev['notes_hit_at_4']:.0%}", "expected note retrieved"),
            (c2, "Candidate recall@5", f"{ev['candidate_recall_at_5']:.0%}", "best songs kept by the retriever"),
            (c3, "Adversarial blocked", f"{ev['adversarial_blocked']:.0%}", "unsafe inputs stopped"),
        ):
            col.markdown(f'<div class="tj stat"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div>'
                         f'<div class="sub">{esc(sub)}</div></div>', unsafe_allow_html=True)
    except Exception as exc:  # the About page must never break the app
        st.caption(f"Evaluation unavailable: {type(exc).__name__}")
    st.caption("Live accuracy per provider: `python scripts/eval_rag.py --live --provider all`. "
               "Unit tests: `pytest` (fake AI clients, runs offline).")

    with st.expander("Architecture diagram source", icon=":material/schema:"):
        path = ROOT / "diagrams" / "architecture.mmd"
        st.caption("Mermaid source in diagrams/architecture.mmd. Paste it into mermaid.live to render it.")
        st.code(path.read_text() if path.exists() else "(diagram not found)", language=None)


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

navigation = st.navigation(
    [
        st.Page(page_assistant, title="AI Assistant", icon=":material/auto_awesome:", url_path="assistant", default=True),
        st.Page(page_explore, title="Explore", icon=":material/tune:", url_path="explore"),
        st.Page(page_insights, title="Insights", icon=":material/insights:", url_path="insights"),
        st.Page(page_catalog, title="Catalog", icon=":material/library_music:", url_path="catalog"),
        st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
    ],
    position="top",
)
navigation.run()

st.markdown(
    f'<div class="tj footer">TuneJudge · content-based scoring over {len(songs):,} real tracks · '
    "audio features &amp; players via Spotify · AI answers checked against the data · no listening history, no tracking</div>",
    unsafe_allow_html=True,
)
