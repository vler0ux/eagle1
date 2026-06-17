# 🚀 Eagle-1 — Pilote Automatique par Apprentissage par Renforcement

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-LunarLander--v3-green)](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.9.0-orange)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Mission AstroDynamics — Développement d'un pilote automatique pour l'atterrisseur lunaire **Eagle-1**, dans le cadre du parcours *AI Engineer* d'OpenClassRooms (Projet 11).

---

## 🌕 Contexte de la mission

AstroDynamics, entreprise spécialisée dans les systèmes autonomes pour l'exploration spatiale, souhaite automatiser les procédures d'atterrissage de ses modules afin d'améliorer la sécurité et d'optimiser la consommation de carburant.

En tant qu'ingénieure Machine Learning junior spécialisée en apprentissage par renforcement, ma mission est de concevoir, entraîner et déployer un agent capable de poser le module **Eagle-1** en douceur sur une zone cible, en utilisant l'environnement simulé [`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) de Gymnasium.

## 🎯 Objectifs

- Explorer l'environnement `LunarLander-v3` (espaces d'observation et d'action, fonction de récompense)
- Établir une performance de référence (baseline) avec des hyperparamètres par défaut
- Optimiser l'agent jusqu'à dépasser de manière stable une récompense moyenne de **200 points** sur 100 épisodes d'évaluation
- Déployer le modèle entraîné via une API, une interface graphique et un tableau de bord interactif

## 🧠 Approche technique

Deux variantes de l'environnement sont explorées en parallèle lors de la phase de baseline :

| Variante | Espace d'action | Algorithme |
|---|---|---|
| Discrète (par défaut) | `Discrete(4)` | DQN |
| Continue | `Box` | PPO |

La librairie [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) est utilisée pour l'entraînement, l'évaluation et la sauvegarde des modèles.

## 📦 Livrables

- [ ] **Notebook Colab/Jupyter** (`.ipynb`) — exploration, choix de l'algorithme, entraînement, optimisation, évaluation
- [ ] **Vidéo** (`.mp4`, 20-30s) — démonstration d'un atterrissage réussi
- [ ] **API** — interface pour interagir avec le modèle entraîné
- [ ] **Interface graphique (GUI)** — visualisation d'une partie jouée par l'agent
- [ ] **Tableau de bord** — suivi interactif des performances de l'agent

## 🗂️ Structure du dépôt

```
eagle1-lunarlander/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── eagle1_lunarlander.ipynb
├── models/              # Modèles SB3 sauvegardés (.zip, ignorés par git)
├── api/                 # Backend (FastAPI/Flask)
├── gui/                 # Interface (Streamlit/Gradio)
└── logs/                # Logs TensorBoard (ignorés par git)
```

## ⚙️ Installation

```bash
# Cloner le dépôt
git clone https://github.com/vler0ux/eagle1-lunarlander.git
cd eagle1-lunarlander

# Créer et activer l'environnement virtuel
python3 -m venv venv-eagle1
source venv-eagle1/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

> 💡 `gymnasium[box2d]` nécessite `swig` comme dépendance système sur Linux :
> ```bash
> sudo apt install swig
> ```

## 🛣️ Roadmap

- [x] Mise en place de l'environnement (venv, dépendances, structure du dépôt)
- [ ] **Étape 1** — Exploration de l'environnement et baseline (DQN discret / PPO continu)
- [ ] **Étape 2** — Optimisation des hyperparamètres (objectif : récompense moyenne > 200)
- [ ] **Étape 3** — Déploiement (API, GUI, tableau de bord)

## 🛠️ Outils & Librairies

- [Gymnasium](https://gymnasium.farama.org/) — environnement de simulation
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — algorithmes RL (DQN, PPO)
- [PyTorch](https://pytorch.org/) — backend de calcul
- [Matplotlib](https://matplotlib.org/) — visualisation des résultats
- [TensorBoard](https://www.tensorflow.org/tensorboard) — suivi des courbes d'entraînement

## 📄 Licence

Ce projet est sous licence [MIT](LICENSE).

---

*Projet réalisé dans le cadre du parcours AI Engineer d'OpenClassRooms — Projet 11.*
