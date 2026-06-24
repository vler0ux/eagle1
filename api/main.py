"""
API Eagle-1 AstroDynamics
=========================
Sert le modèle PPO entraîné sur LunarLander-v3 (variante continue).

Conformément au brief de mission : toute la logique RL (chargement du modèle,
inférence, simulation de l'environnement) vit ici, côté backend. La GUI et le
futur tableau de bord ne font qu'appeler ces endpoints — ils ne doivent jamais
charger Stable-Baselines3 ni le modèle eux-mêmes.

Lancement local :
    uvicorn main:app --reload --port 8000

Documentation interactive générée automatiquement : http://localhost:8000/docs
"""
import json
import time
from pathlib import Path
from typing import List, Optional

import gymnasium as gym
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# À adapter si tu choisis l'autre run (ppo_lunarlander_400k_gamma0999_run2)
MODEL_PATH = Path(__file__).parent / "models" / "ppo_lunarlander_400k_gamma0999.zip"
LOG_PATH = Path(__file__).parent / "episodes_log.jsonl"
SUCCESS_THRESHOLD = 200.0  # seuil de réussite défini dans le brief de mission
ENV_ID = "LunarLander-v3"

app = FastAPI(
    title="Eagle-1 AstroDynamics API",
    description="Interface pour interagir avec le pilote automatique d'Eagle-1 (PPO).",
    version="1.0.0",
)

model: Optional[PPO] = None


@app.on_event("startup")
def load_model() -> None:
    """Charge le modèle une seule fois, au démarrage du serveur."""
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Place le fichier .zip du modèle entraîné dans le dossier 'models/' "
            "à côté de ce fichier, ou ajuste MODEL_PATH."
        )
    model = PPO.load(MODEL_PATH)
    print(f"[Eagle-1 API] Modèle chargé : {MODEL_PATH.name}")


# --------------------------------------------------------------------------
# Schémas (validation automatique des requêtes/réponses)
# --------------------------------------------------------------------------

class Observation(BaseModel):
    """Vecteur d'observation LunarLander-v3 : x, y, vx, vy, angle, v_angle,
    contact_jambe_gauche, contact_jambe_droite."""
    state: List[float] = Field(..., min_length=8, max_length=8)


class ActionResponse(BaseModel):
    action: List[float]  # [poussée moteur principal, poussée latérale] dans [-1, 1]


class EpisodeStep(BaseModel):
    step: int
    observation: List[float]
    action: List[float]
    reward: float


class EpisodeResponse(BaseModel):
    steps: List[EpisodeStep]
    total_reward: float
    length: int
    success: bool
    crashed: bool
    max_abs_angle_deg: float
    mean_main_thrust: float
    mean_lateral_thrust: float


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """Vérifie que l'API tourne et que le modèle est chargé."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/play", response_model=ActionResponse)
def play(obs: Observation) -> ActionResponse:
    """
    Reçoit un état et renvoie l'action choisie par le modèle entraîné.
    N'exécute aucune simulation : c'est une simple inférence, conforme à la
    spécification du brief de mission ("endpoint /play acceptant un état et
    renvoyant une action").
    """
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    state = np.array(obs.state, dtype=np.float32)
    action, _ = model.predict(state, deterministic=True)
    return ActionResponse(action=action.tolist())


@app.post("/episode", response_model=EpisodeResponse)
def play_episode(seed: Optional[int] = None) -> EpisodeResponse:
    """
    Joue un épisode complet en interne (environnement + modèle, tous deux
    côté backend) et renvoie la trajectoire complète. Pensé pour être
    consommé par la GUI (animation) et pour alimenter les logs du dashboard.
    """
    if model is None:
        raise HTTPException(503, "Modèle non chargé")

    env = gym.make(ENV_ID, continuous=True)
    observation, _ = env.reset(seed=seed)

    steps: List[EpisodeStep] = []
    total_reward = 0.0
    terminated = truncated = False
    step_count = 0

    while not (terminated or truncated):
        action, _ = model.predict(observation, deterministic=True)
        next_observation, reward, terminated, truncated, _ = env.step(action)

        steps.append(EpisodeStep(
            step=step_count,
            observation=observation.tolist(),
            action=action.tolist(),
            reward=float(reward),
        ))

        observation = next_observation
        total_reward += float(reward)
        step_count += 1

    env.close()

    success = total_reward > SUCCESS_THRESHOLD
    # Un crash se traduit par la pénalité terminale de -100 définie par l'environnement
    crashed = bool(steps) and steps[-1].reward <= -90

    angles = [abs(s.observation[4]) for s in steps]
    main_thrusts = [s.action[0] for s in steps if s.action[0] > 0]
    lateral_thrusts = [abs(s.action[1]) for s in steps]

    max_abs_angle_deg = float(np.degrees(max(angles))) if angles else 0.0
    mean_main_thrust = float(np.mean(main_thrusts)) if main_thrusts else 0.0
    mean_lateral_thrust = float(np.mean(lateral_thrusts)) if lateral_thrusts else 0.0

    _log_episode(
        total_reward, step_count, success, crashed,
        max_abs_angle_deg, mean_main_thrust, mean_lateral_thrust,
    )

    return EpisodeResponse(
        steps=steps,
        total_reward=total_reward,
        length=step_count,
        success=success,
        crashed=crashed,
        max_abs_angle_deg=max_abs_angle_deg,
        mean_main_thrust=mean_main_thrust,
        mean_lateral_thrust=mean_lateral_thrust,
    )


def _log_episode(
    total_reward: float, length: int, success: bool, crashed: bool,
    max_abs_angle_deg: float, mean_main_thrust: float, mean_lateral_thrust: float,
) -> None:
    """Ajoute une ligne au fichier de logs (format JSONL) utilisé par le tableau de bord."""
    entry = {
        "timestamp": time.time(),
        "total_reward": total_reward,
        "length": length,
        "success": success,
        "crashed": crashed,
        "max_abs_angle_deg": max_abs_angle_deg,
        "mean_main_thrust": mean_main_thrust,
        "mean_lateral_thrust": mean_lateral_thrust,
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
