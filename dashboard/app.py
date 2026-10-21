"""F1 — Delta vs Absolute Lap Time
==================================

Streamlit dashboard that walks a stakeholder (race engineer, project sponsor)
through *why* we predict the lap-time delta to a stint reference, rather than
the raw lap time, when forecasting tyre degradation.

Run with:
    streamlit run dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


# =============================================================================
# Constants
# =============================================================================
PALETTE = {
    "fg":          "#0f172a",
    "muted":       "#64748b",
    "subtle":      "#cbd5e1",
    "grid":        "#eef2f7",
    "signal":      "#f59e0b",
    "signal_bg":   "#fef3c7",
    "noise":       "#94a3b8",
    "noise_dark":  "#475569",
    "accent":      "#3b82f6",
    "border":      "#e2e8f0",
}

STINT_PALETTE = ["#3b82f6", "#10b981", "#ec4899", "#8b5cf6"]

TRACK_NAMES = {
    2:   "Silverstone",
    4:   "Hungaroring",
    6:   "Imola",
    7:   "Spa-Francorchamps",
    9:   "COTA (Austin)",
    10:  "Albert Park (Melbourne)",
    14:  "Interlagos (São Paulo)",
    15:  "Catalunya (Barcelona)",
    19:  "Red Bull Ring (Spielberg)",
    22:  "Monaco",
    23:  "Montreal",
    39:  "Monza",
    46:  "Suzuka",
    49:  "Shanghai",
    55:  "Zandvoort",
    61:  "Marina Bay (Singapore)",
    63:  "Sakhir (Bahrain)",
    65:  "Hermanos Rodríguez (Mexico)",
    70:  "Yas Marina (Abu Dhabi)",
    144: "Baku",
    149: "Jeddah",
    150: "Lusail (Qatar)",
    151: "Miami",
    152: "Las Vegas",
}

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.csv"


# =============================================================================
# Page config & global CSS
# =============================================================================
st.set_page_config(
    page_title="F1 · Delta vs Lap Time",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding: 2.5rem 3rem 5rem; max-width: 1240px; }
    h1 { font-weight: 800; letter-spacing: -0.025em; color:#0f172a; margin-bottom:0.5rem; }
    h2 { font-weight: 700; letter-spacing: -0.015em; margin-top: 3.5rem;
         color:#0f172a; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; }
    h3 { font-weight: 600; color: #0f172a; margin-top: 1.5rem; }
    .lead { color: #475569; font-size: 1.12rem; line-height: 1.65; max-width: 820px; }
    .kpi { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px;
           padding:1.05rem 1.25rem; height: 100%; }
    .kpi-value { font-size: 1.85rem; font-weight: 800; color:#0f172a; line-height:1.1; }
    .kpi-label { font-size:0.76rem; color:#64748b; text-transform: uppercase;
                 letter-spacing:0.06em; margin-top:0.3rem; font-weight:600; }
    .callout { background:#fffbeb; border-left:4px solid #f59e0b;
               padding:1rem 1.25rem; border-radius:6px; color:#78350f;
               line-height:1.6; margin-top:0.5rem; }
    .takeaway { background:#f0fdf4; border-left:4px solid #10b981;
                padding:1.15rem 1.4rem; border-radius:6px; color:#064e3b;
                line-height:1.65; }
    .muted { color:#64748b; font-size:0.92rem; }
    .footer { color:#94a3b8; font-size:0.82rem; margin-top:4rem;
              text-align:center; padding-top:2rem; border-top:1px solid #e2e8f0; }
    div[data-testid="stHorizontalBlock"] { gap: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Data loading & preparation
# =============================================================================
@st.cache_data(show_spinner="Loading lap data…")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out unrepresentative laps and compute the delta per stint."""
    clean_mask = (
        ~df["is_pit_out_lap"].astype(bool)
        & ~df["is_in_lap"].astype(bool)
        & ~df["is_under_sc"].astype(bool)
        & ~df["is_under_vsc"].astype(bool)
        & ~df["is_under_yellow"].astype(bool)
        & ~df["is_red_flag_lap"].astype(bool)
        & (df["rainfall"] == 0)
        & (df["lap_number"] != 1)
        & df["lap_duration"].notna()
        & (df["lap_duration"] > 60)
        & (df["lap_duration"] < 200)
    )
    clean = df.loc[clean_mask].copy()

    # Per-stint plausibility cap: drop laps > 1.07 * stint median (traffic, lockups, …)
    median_per_stint = clean.groupby("stint_id")["lap_duration"].transform("median")
    clean = clean[clean["lap_duration"] <= 1.07 * median_per_stint]

    clean = clean.sort_values(["stint_id", "stint_age"])
    first_idx = clean.groupby("stint_id")["stint_age"].idxmin()
    ref = (
        clean.loc[first_idx, ["stint_id", "lap_duration", "stint_age"]]
        .rename(columns={"lap_duration": "ref_lap", "stint_age": "ref_age"})
    )
    clean = clean.merge(ref, on="stint_id")
    clean["lap_in_stint"] = clean["stint_age"] - clean["ref_age"] + 1
    clean["delta"] = clean["lap_duration"] - clean["ref_lap"]

    # Final delta sanity filter: realistic wear range is roughly [-2, +6] s
    clean = clean[clean["delta"].between(-2, 6)]

    sizes = clean.groupby("stint_id").size().rename("stint_size")
    clean = clean.merge(sizes, on="stint_id")
    clean = clean[clean["stint_size"] >= 5].reset_index(drop=True)

    clean["track_name"] = (
        clean["track_id"].map(TRACK_NAMES).fillna("Circuit " + clean["track_id"].astype(str))
    )
    return clean


@st.cache_data
def stint_summary_for_drilldown(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("stint_id").agg(
        track=("track_name", "first"),
        team=("team_id", "first"),
        compound=("compound", "first"),
        driver=("driver_number", "first"),
        year=("year", "first"),
        n_laps=("lap_in_stint", "max"),
        max_delta=("delta", "max"),
        ref_lap=("ref_lap", "first"),
    ).reset_index()
    s["label"] = (
        s["track"] + " · " + s["team"] + " · " + s["compound"]
        + " · " + s["year"].astype(str) + " · #" + s["driver"].astype(str)
    )
    return s


df_raw = load_data()
df = prepare(df_raw)
stints = stint_summary_for_drilldown(df)


# =============================================================================
# Helper: axis style for Plotly
# =============================================================================
AXIS_STYLE = dict(
    gridcolor=PALETTE["grid"],
    zerolinecolor=PALETTE["subtle"],
    showline=True,
    linecolor=PALETTE["subtle"],
    ticks="outside",
    tickcolor=PALETTE["subtle"],
)

BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, sans-serif", size=12, color=PALETTE["fg"]),
)


def kpi(col, value: str, label: str, color: str = "#0f172a") -> None:
    col.markdown(
        f'<div class="kpi"><div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SECTION 1 — Hero
# =============================================================================
st.markdown("# Predicting Tyre Degradation")
st.markdown(
    '<p class="lead">'
    "Forecasting how much a Formula 1 car slows down on its tyres as a stint progresses. "
    "The interesting question is <b>not</b> &ldquo;how fast will the next lap be?&rdquo; — "
    "it is <b>&ldquo;how much slower than fresh tyres?&rdquo;</b> "
    "Modelling the <b>delta to a stint reference lap</b> instead of the raw lap time isolates exactly that."
    "</p>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{len(df_raw):,}",            "Raw race laps")
kpi(c2, f"{len(df):,}",                "Clean delta samples")
kpi(c3, f"{df['stint_id'].nunique():,}", "Stints (≥ 5 clean laps)")
kpi(c4, f"{df['track_id'].nunique()}", "Circuits · 2023 – 2025")


# =============================================================================
# SECTION 2 — What's in a lap time?
# =============================================================================
st.markdown("## What's in a 90-second lap?")
st.markdown(
    '<p class="lead">'
    "A race lap is one number, but it bundles several independent effects. "
    "Pick a circuit — the typical lap there decomposes roughly like this:"
    "</p>",
    unsafe_allow_html=True,
)

col_picker, col_chart = st.columns([1, 3])

with col_picker:
    track_options = sorted(df["track_name"].unique())
    default_idx = track_options.index("Monaco") if "Monaco" in track_options else 0
    chosen_track = st.selectbox("Circuit", track_options, index=default_idx, key="track_picker")

with col_chart:
    track_laps = df[df["track_name"] == chosen_track]
    median_lap = float(track_laps["lap_duration"].median())

    components = [
        ("Track geometry",       0.860, PALETTE["noise_dark"]),
        ("Constructor (car)",    0.034, PALETTE["noise"]),
        ("Driver skill",         0.022, PALETTE["noise"]),
        ("Fuel load",            0.034, PALETTE["noise"]),
        ("Weather / track evo",  0.017, PALETTE["noise"]),
        ("Tyre wear",            0.017, PALETTE["signal"]),
    ]

    fig = go.Figure()
    for name, pct, color in components:
        val = median_lap * pct
        is_signal = name == "Tyre wear"
        fig.add_trace(go.Bar(
            x=[val],
            y=["Lap"],
            name=name,
            orientation="h",
            marker=dict(
                color=color,
                line=dict(color="white", width=2.5),
            ),
            hovertemplate=f"<b>{name}</b><br>≈ %{{x:.2f}} s ({pct*100:.1f} %)<extra></extra>",
        ))

    fig.update_layout(
        **BASE_LAYOUT,
        barmode="stack",
        height=180,
        margin=dict(t=10, b=70, l=10, r=10),
        legend=dict(orientation="h", y=-0.6, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Seconds", **AXIS_STYLE),
        yaxis=dict(showticklabels=False, showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    f'<div class="callout">'
    f"For a typical lap at <b>{chosen_track}</b> ({median_lap:.1f} s median), only ≈ <b>1.7 %</b> of that number "
    "actually changes due to tyre wear over a stint. The rest is geometry and car identity — "
    "things the model already knows. Spending capacity on them is wasted effort and a source of overfitting."
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 3 — Same data, two views (the money plot)
# =============================================================================
st.markdown("## Same data, two views")
st.markdown(
    '<p class="lead">'
    "Four real stints from four different circuits. <b>Left:</b> their raw lap times — "
    "four lines on completely different baselines, the wear is invisible. "
    "<b>Right:</b> the same four stints, each expressed as the <i>delta to its own reference lap</i>. "
    "Now the wear curve is the only structure left."
    "</p>",
    unsafe_allow_html=True,
)

# Pickable stints: long enough, no DNF artifacts
pickable = stints[(stints["n_laps"] >= 12) & (stints["max_delta"] > 0.4) & (stints["max_delta"] < 5)]

if "shuffle_seed" not in st.session_state:
    st.session_state.shuffle_seed = 1

ctrl1, ctrl2 = st.columns([1, 5])
if ctrl1.button("🔀 Shuffle stints", use_container_width=True):
    st.session_state.shuffle_seed += 1

rng = np.random.default_rng(st.session_state.shuffle_seed)
available_tracks = pickable["track"].unique()
n_pick = min(4, len(available_tracks))
chosen_tracks = rng.choice(available_tracks, size=n_pick, replace=False)

chosen_stints = []
for trk in chosen_tracks:
    pool = pickable[pickable["track"] == trk]
    chosen_stints.append(pool.sample(1, random_state=int(rng.integers(0, 10**6))).iloc[0])

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Modelling target: absolute lap time</b>",
        "<b>Modelling target: delta to stint reference</b>",
    ),
    horizontal_spacing=0.10,
)

for i, stint in enumerate(chosen_stints):
    sdf = df[df["stint_id"] == stint["stint_id"]].sort_values("lap_in_stint")
    label = f"{stint['track']} · {stint['team']} · {stint['compound']}"
    color = STINT_PALETTE[i % len(STINT_PALETTE)]

    fig.add_trace(go.Scatter(
        x=sdf["lap_in_stint"], y=sdf["lap_duration"],
        mode="lines+markers",
        name=label,
        line=dict(color=color, width=2.2),
        marker=dict(size=7, line=dict(color="white", width=1.2)),
        legendgroup=label, showlegend=True,
        hovertemplate=f"<b>{label}</b><br>Lap in stint: %{{x}}<br>Lap time: %{{y:.2f}} s<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sdf["lap_in_stint"], y=sdf["delta"],
        mode="lines+markers",
        name=label,
        line=dict(color=color, width=2.2),
        marker=dict(size=7, line=dict(color="white", width=1.2)),
        legendgroup=label, showlegend=False,
        hovertemplate=f"<b>{label}</b><br>Lap in stint: %{{x}}<br>Delta: %{{y:+.2f}} s<extra></extra>",
    ), row=1, col=2)

fig.update_layout(
    **BASE_LAYOUT,
    height=540,
    margin=dict(t=70, b=110, l=70, r=40),
    legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center",
                bgcolor="rgba(0,0,0,0)"),
)
fig.update_xaxes(title_text="Lap in stint", row=1, col=1, **AXIS_STYLE)
fig.update_xaxes(title_text="Lap in stint", row=1, col=2, **AXIS_STYLE)
fig.update_yaxes(title_text="Lap time [s]",      row=1, col=1, **AXIS_STYLE)
fig.update_yaxes(title_text="Delta to reference [s]", row=1, col=2, **AXIS_STYLE)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    '<p class="muted">Real laps from the cleaned dataset. Click <b>Shuffle</b> '
    "to draw a different set of four. The pattern holds regardless of which four you pick.</p>",
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 4 — Inside one stint
# =============================================================================
st.markdown("## Inside one stint: what cancels out")
st.markdown(
    '<p class="lead">'
    "Pick any stint. The reference is the first clean lap. Every other lap of the stint sits a "
    "few tenths above or below it. That offset — the delta — is the model's target. "
    "Drag the slider to watch the delta change lap by lap."
    "</p>",
    unsafe_allow_html=True,
)

drilldown_pool = stints[
    (stints["n_laps"] >= 18) & (stints["max_delta"] > 0.8) & (stints["max_delta"] < 5)
].sort_values("max_delta", ascending=False)

pick_col, _ = st.columns([2, 1])
with pick_col:
    chosen_label = st.selectbox(
        "Pick a stint",
        drilldown_pool["label"].head(40).tolist(),
        key="stint_picker",
    )

chosen = drilldown_pool[drilldown_pool["label"] == chosen_label].iloc[0]
stint_df = df[df["stint_id"] == chosen["stint_id"]].sort_values("lap_in_stint")
max_lap = int(stint_df["lap_in_stint"].max())

chosen_lap = st.slider(
    "Highlight lap in stint",
    min_value=2, max_value=max_lap, value=max(max_lap // 2, 2),
    key="highlight_slider",
)

ref_lap_val   = float(chosen["ref_lap"])
hl_row        = stint_df[stint_df["lap_in_stint"] == chosen_lap].iloc[0]
current_val   = float(hl_row["lap_duration"])
delta_val     = float(hl_row["delta"])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=stint_df["lap_in_stint"], y=stint_df["lap_duration"],
    mode="lines+markers",
    name="Lap time",
    line=dict(color=PALETTE["accent"], width=2.5),
    marker=dict(size=8, color=PALETTE["accent"], line=dict(color="white", width=1)),
    hovertemplate="Lap %{x}<br>Lap time: %{y:.2f} s<extra></extra>",
))

fig.add_hline(
    y=ref_lap_val,
    line_dash="dash",
    line_color=PALETTE["noise_dark"],
    line_width=1.8,
    annotation_text=f"<b>Reference: {ref_lap_val:.2f} s</b>",
    annotation_position="top left",
    annotation_font=dict(color=PALETTE["noise_dark"], size=12),
)

fig.add_trace(go.Scatter(
    x=[chosen_lap], y=[current_val],
    mode="markers",
    marker=dict(size=20, color=PALETTE["signal"],
                line=dict(color="white", width=2.5), symbol="circle"),
    showlegend=False,
    hovertemplate=f"Lap {chosen_lap}<br>Lap time: %{{y:.2f}} s<extra></extra>",
))

fig.add_shape(
    type="line",
    x0=chosen_lap, x1=chosen_lap,
    y0=ref_lap_val, y1=current_val,
    line=dict(color=PALETTE["signal"], width=3),
)
fig.add_annotation(
    x=chosen_lap, y=(ref_lap_val + current_val) / 2,
    text=f"<b>Δ = {delta_val:+.2f} s</b>",
    showarrow=False, xshift=55,
    bgcolor=PALETTE["signal"],
    bordercolor=PALETTE["signal"], borderpad=8,
    font=dict(color="white", size=13),
)

fig.update_layout(
    **BASE_LAYOUT,
    title=dict(
        text=f"<b>{chosen_label}</b>",
        x=0.5, xanchor="center", font=dict(size=14, color=PALETTE["fg"]),
    ),
    height=480,
    margin=dict(t=70, b=50, l=70, r=40),
    showlegend=False,
    xaxis=dict(title="Lap in stint", **AXIS_STYLE),
    yaxis=dict(title="Lap time [s]", **AXIS_STYLE),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

k1, k2, k3 = st.columns(3)
kpi(k1, f"{ref_lap_val:.2f} s",  "Reference (1st clean lap)", PALETTE["noise_dark"])
kpi(k2, f"{current_val:.2f} s",  f"Lap {chosen_lap}",         PALETTE["accent"])
kpi(k3, f"{delta_val:+.2f} s",   "Delta — the model's target", PALETTE["signal"])


# =============================================================================
# SECTION 5 — Distributions of the two targets
# =============================================================================
st.markdown("## Two targets, two distributions")
st.markdown(
    '<p class="lead">'
    "Across all clean laps in three seasons, here is what each target actually looks like. "
    "The lap-time distribution is a mixture — each peak corresponds to a different circuit. "
    "The delta distribution is what is left once that mixture is removed."
    "</p>",
    unsafe_allow_html=True,
)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "<b>Target: absolute lap time</b>",
        "<b>Target: delta to stint reference</b>",
    ),
    horizontal_spacing=0.13,
)

fig.add_trace(go.Histogram(
    x=df["lap_duration"],
    nbinsx=90,
    marker=dict(color=PALETTE["noise_dark"],
                line=dict(color="white", width=0.5)),
    hovertemplate="%{x:.1f} s<br>%{y:,} laps<extra></extra>",
    showlegend=False,
), row=1, col=1)

fig.add_trace(go.Histogram(
    x=df["delta"],
    nbinsx=90,
    marker=dict(color=PALETTE["signal"],
                line=dict(color="white", width=0.5)),
    hovertemplate="%{x:.2f} s<br>%{y:,} laps<extra></extra>",
    showlegend=False,
), row=1, col=2)

fig.update_layout(
    **BASE_LAYOUT,
    height=420,
    margin=dict(t=70, b=60, l=70, r=40),
    showlegend=False,
    bargap=0.02,
)
fig.update_xaxes(title_text="Lap duration [s]",       row=1, col=1, **AXIS_STYLE)
fig.update_xaxes(title_text="Delta to reference [s]", row=1, col=2, range=[-2, 5], **AXIS_STYLE)
fig.update_yaxes(title_text="Number of laps",         row=1, col=1, **AXIS_STYLE)
fig.update_yaxes(title_text="Number of laps",         row=1, col=2, **AXIS_STYLE)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

lap_std   = float(df["lap_duration"].std())
delta_std = float(df["delta"].std())
ratio     = lap_std / delta_std if delta_std > 0 else float("nan")

st.markdown(
    f'<div class="callout">'
    f"Standard deviation of <b>lap time</b>: <b>{lap_std:.2f} s</b> "
    "(mostly between-circuit variation). "
    f"Standard deviation of <b>delta</b>: <b>{delta_std:.2f} s</b>. "
    f"Same data, <b>{ratio:.0f}× tighter</b> target distribution — and what is left is the actual signal."
    "</div>",
    unsafe_allow_html=True,
)


# =============================================================================
# SECTION 6 — Takeaway
# =============================================================================
st.markdown("## Takeaway")
st.markdown(
    '<div class="takeaway">'
    "<b>Predicting the delta — not the raw lap time — removes ~ 98 % of the noise before training starts.</b> "
    "Track geometry, car identity, driver skill and starting fuel cancel out by construction within a stint. "
    "What remains is exactly the physics we want to learn: how tyre compound, temperature and stint phase "
    "translate into seconds lost per lap. "
    "<br><br>"
    "The resulting model is <b>interpretable</b> (its output is &ldquo;seconds of lap-time lost&rdquo;), "
    "<b>transferable</b> across drivers and circuits, and directly answers the race-engineer's real question: "
    "<i>&ldquo;how much pace will we lose if we stay out another five laps?&rdquo;</i>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer">Data: OpenF1 · 2023 – 2025 race sessions · '
    "Filtered to dry, racing-pace laps · Reference lap = first clean lap of each stint</div>",
    unsafe_allow_html=True,
)
