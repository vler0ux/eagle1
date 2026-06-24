"""
GUI Eagle-1 AstroDynamics
=========================
Visualise une partie jouée par le pilote automatique (modèle PPO).

Conformément au brief de mission, cette interface ne contient AUCUNE logique
RL : elle se contente d'appeler l'API (/episode) et d'animer la trajectoire
renvoyée. Le modèle et l'environnement Gymnasium ne sont jamais chargés ici.

Lancement local (API déjà démarrée sur le port 8000) :
    streamlit run app.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"
FRAME_DELAY = 0.03  # secondes entre deux frames à vitesse normale (≈33 fps)

# Géométrie approximative de la zone d'atterrissage (helipad), pour repère visuel
PAD_HALF_WIDTH = 0.2


def draw_frame(ax, observation, action, step, reward):
    """Dessine un état du lander : position, orientation, contact au sol, flammes."""
    x, y, vx, vy, angle, v_angle, leg_left, leg_right = observation
    main_thrust, lateral_thrust = action

    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Sol et zone d'atterrissage
    ax.axhline(0, color="dimgray", linewidth=2, zorder=1)
    ax.fill_betweenx([0, 0.02], -PAD_HALF_WIDTH, PAD_HALF_WIDTH, color="gold", alpha=0.4, zorder=1)

    # Corps du lander (triangle orienté selon `angle`)
    body_size = 0.09
    body_pts = np.array([
        [0, body_size],
        [-body_size * 0.7, -body_size * 0.5],
        [body_size * 0.7, -body_size * 0.5],
    ])
    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    body_world = body_pts @ rotation.T + np.array([x, y])

    en_contact = bool(leg_left) or bool(leg_right)
    body_color = "mediumseagreen" if en_contact else "silver"
    ax.fill(body_world[:, 0], body_world[:, 1], color=body_color, zorder=3)

    # Jambes (vertes si contact au sol)
    leg_base = np.array([0, -body_size * 0.5])
    for sign, contact in [(-1, leg_left), (1, leg_right)]:
        leg_tip = leg_base + np.array([sign * body_size * 0.9, -body_size * 0.4])
        leg_world = np.array([leg_base, leg_tip]) @ rotation.T + np.array([x, y])
        color = "limegreen" if contact else "dimgray"
        ax.plot(leg_world[:, 0], leg_world[:, 1], color=color, linewidth=2, zorder=2)

    # Flammes (indicatives, proportionnelles à l'intensité de l'action)
    if main_thrust > 0:
        flame_pts = np.array([
            [-body_size * 0.3, -body_size * 0.5],
            [body_size * 0.3, -body_size * 0.5],
            [0, -body_size * (0.5 + 0.8 * main_thrust)],
        ])
        flame_world = flame_pts @ rotation.T + np.array([x, y])
        ax.fill(flame_world[:, 0], flame_world[:, 1], color="orangered", alpha=0.8, zorder=2)

    if abs(lateral_thrust) > 0.5:
        side = 1 if lateral_thrust > 0 else -1
        flame_pts = np.array([
            [side * body_size * 0.7, -body_size * 0.1],
            [side * body_size * 0.7, body_size * 0.1],
            [side * body_size * 1.4, 0],
        ])
        flame_world = flame_pts @ rotation.T + np.array([x, y])
        ax.fill(flame_world[:, 0], flame_world[:, 1], color="deepskyblue", alpha=0.8, zorder=2)

    ax.set_title(
        f"pas {step}  ·  angle={np.degrees(angle):.0f}°  ·  reward={reward:+.1f}",
        fontsize=10,
    )


def play_episode_request():
    response = requests.post(f"{API_URL}/episode", timeout=60)
    response.raise_for_status()
    return response.json()


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

st.set_page_config(page_title="Eagle-1 — Visualisation d'un atterrissage", layout="centered")
st.title("Eagle-1 — Visualisation d'un atterrissage")
st.caption(
    "L'action est décidée par le modèle PPO côté API. Cette interface se contente "
    "d'afficher la partie jouée — aucune logique RL ici."
)

speed = st.slider("Vitesse de lecture", min_value=0.5, max_value=3.0, value=1.0, step=0.5)

if st.button("Lancer une nouvelle partie", type="primary"):
    try:
        with st.spinner("L'API joue l'épisode..."):
            st.session_state["episode"] = play_episode_request()
    except requests.exceptions.RequestException as exc:
        st.error(f"Impossible de contacter l'API ({API_URL}). Est-elle bien lancée ? Détail : {exc}")

if "episode" in st.session_state:
    episode = st.session_state["episode"]
    steps = episode["steps"]

    fig, ax = plt.subplots(figsize=(5, 5))
    frame_placeholder = st.empty()

    for s in steps:
        draw_frame(ax, s["observation"], s["action"], s["step"], s["reward"])
        frame_placeholder.pyplot(fig)
        time.sleep(FRAME_DELAY / speed)

    plt.close(fig)

    st.subheader("Résultat de la partie")
    col1, col2, col3 = st.columns(3)
    col1.metric("Récompense totale", f"{episode['total_reward']:.1f}")
    col2.metric("Durée", f"{episode['length']} pas")
    col3.metric("Atterrissage réussi", "Oui" if episode["success"] else "Non")
