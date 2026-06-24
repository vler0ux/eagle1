"""
Tableau de bord Eagle-1 AstroDynamics
======================================
Affiche les métriques agrégées sur l'ensemble des épisodes journalisés par
l'API (api/episodes_log.jsonl). Ne contient aucune logique RL — lecture de
logs uniquement, conformément à la séparation backend/frontend du brief.

Lancement local :
    streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

LOG_PATH = Path(__file__).parent.parent / "api" / "episodes_log.jsonl"
SUCCESS_THRESHOLD = 200.0

st.set_page_config(page_title="Eagle-1 — Tableau de bord", layout="wide")
st.title("Eagle-1 — Tableau de bord de performance")
st.caption(
    "Métriques agrégées sur les épisodes journalisés par l'API. "
    "Lecture de logs uniquement — aucune inférence ni simulation ici."
)


@st.cache_data(ttl=5)
def load_episodes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    df["episode"] = range(1, len(df) + 1)
    return df


df = load_episodes(LOG_PATH)

if st.button("Recharger les données"):
    st.cache_data.clear()
    df = load_episodes(LOG_PATH)

if df.empty:
    st.warning(
        f"Aucun épisode trouvé dans `{LOG_PATH}`. Joue quelques parties via la "
        "GUI, ou lance `python scripts/seed_episodes.py --n 30` pour peupler "
        "rapidement un échantillon représentatif."
    )
    st.stop()

# Colonnes optionnelles (absentes ou incomplètes si le fichier mélange d'anciens
# logs, avant l'ajout des métriques de circonstance, avec des logs récents).
# On force les types pour éviter les colonnes "object" issues d'un mélange
# True/False/NaN, sur lesquelles les opérateurs logiques (~) échouent.
if "crashed" not in df.columns:
    df["crashed"] = False
df["crashed"] = df["crashed"].fillna(False).astype(bool)

df["success"] = df["success"].fillna(False).astype(bool)

for col in ["max_abs_angle_deg", "mean_main_thrust", "mean_lateral_thrust"]:
    if col not in df.columns:
        df[col] = float("nan")
    df[col] = pd.to_numeric(df[col], errors="coerce")

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------

n_episodes = len(df)
mean_reward = df["total_reward"].mean()
std_reward = df["total_reward"].std()
success_rate = df["success"].mean() * 100
crash_rate = df["crashed"].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Épisodes journalisés", n_episodes)
col2.metric("Récompense moyenne", f"{mean_reward:.1f}")
col3.metric("Écart-type", f"{std_reward:.1f}")
col4.metric("Taux de réussite", f"{success_rate:.0f}%")
col5.metric("Taux de crash", f"{crash_rate:.0f}%")

if n_episodes < 10:
    st.info(
        "Moins de 10 épisodes journalisés : ces statistiques ne sont pas encore "
        "représentatives. Lance `scripts/seed_episodes.py` pour un échantillon "
        "plus fiable."
    )

st.divider()

# --------------------------------------------------------------------------
# Récompense par épisode
# --------------------------------------------------------------------------

st.subheader("Récompense par épisode")

df["reward_rolling_mean"] = df["total_reward"].rolling(window=5, min_periods=1).mean()

fig1, ax1 = plt.subplots(figsize=(7, 3), dpi=140)
ax1.plot(df["episode"], df["total_reward"], "o-", color="silver", alpha=0.6, markersize=3, linewidth=1, label="Récompense brute")
ax1.plot(df["episode"], df["reward_rolling_mean"], color="steelblue", linewidth=2, label="Moyenne glissante (5)")
ax1.axhline(SUCCESS_THRESHOLD, color="seagreen", linestyle="--", linewidth=1, label="Seuil de réussite (200)")
ax1.set_xlabel("Épisode", fontsize=9)
ax1.set_ylabel("Récompense totale", fontsize=9)
ax1.tick_params(labelsize=8)
ax1.legend(loc="lower right", fontsize=7)
ax1.grid(alpha=0.3)
fig1.tight_layout()
st.pyplot(fig1, use_container_width=False)
plt.close(fig1)

st.divider()

# --------------------------------------------------------------------------
# Issue de l'épisode selon les circonstances
# --------------------------------------------------------------------------

st.subheader("Issue de l'épisode selon l'angle maximal atteint")
st.caption(
    "Chaque point est un épisode. Un angle maximal élevé pendant la descente "
    "est souvent corrélé à une perte de contrôle — ce graphique permet de "
    "vérifier si c'est bien le cas pour ce modèle."
)

df_angle = df.dropna(subset=["max_abs_angle_deg"])
scatter_col, bar_col = st.columns(2)

with scatter_col:
    st.markdown("**Angle max. vs récompense**")
    if df_angle.empty:
        st.info(
            "Pas de données d'angle disponibles (logs générés avant l'ajout des "
            "métriques de circonstance dans l'API)."
        )
    else:
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5), dpi=140)
        colors = df_angle["success"].map({True: "seagreen", False: "indianred"})
        ax2.scatter(df_angle["max_abs_angle_deg"], df_angle["total_reward"], c=colors, alpha=0.7, s=20)
        ax2.set_xlabel("Angle maximal (degrés)", fontsize=8)
        ax2.set_ylabel("Récompense totale", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        ax2.grid(alpha=0.3)

        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="seagreen", markersize=6, label="Réussi"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="indianred", markersize=6, label="Échoué"),
        ]
        ax2.legend(handles=handles, loc="upper right", fontsize=7)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=False)
        plt.close(fig2)

with bar_col:
    st.markdown("**Répartition des issues**")
    outcome_counts = pd.Series({
        "Réussi": ((df["success"]) & (~df["crashed"])).sum(),
        "Crash": df["crashed"].sum(),
        "Échec (timeout/hors-zone)": ((~df["success"]) & (~df["crashed"])).sum(),
    })

    fig3, ax3 = plt.subplots(figsize=(4.5, 3.5), dpi=140)
    colors_bar = ["seagreen", "indianred", "goldenrod"]
    ax3.barh(outcome_counts.index, outcome_counts.values, color=colors_bar)
    ax3.set_xlabel("Nombre d'épisodes", fontsize=8)
    ax3.tick_params(labelsize=7)
    for i, v in enumerate(outcome_counts.values):
        ax3.text(v, i, f" {v}", va="center", fontsize=8)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=False)
    plt.close(fig3)

st.divider()

with st.expander("Voir les données brutes"):
    st.dataframe(df.drop(columns=["reward_rolling_mean"]), use_container_width=True)
