"""
Génère un échantillon d'épisodes en appelant l'API à plusieurs reprises.
Utile pour peupler api/episodes_log.jsonl avant de consulter le tableau de bord
(une poignée d'épisodes ne suffit pas à juger du taux de réussite réel de l'agent).

Usage :
    python seed_episodes.py --n 30
"""
import argparse
import sys

import requests

API_URL = "http://127.0.0.1:8000"


def main(n: int) -> None:
    successes = 0
    for i in range(1, n + 1):
        try:
            response = requests.post(f"{API_URL}/episode", timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            print(f"Erreur à l'épisode {i} : {exc}", file=sys.stderr)
            continue

        episode = response.json()
        successes += int(episode["success"])
        print(
            f"[{i}/{n}] reward={episode['total_reward']:.1f}  "
            f"length={episode['length']}  success={episode['success']}"
        )

    print(f"\nTerminé : {successes}/{n} épisodes réussis "
          f"({100 * successes / n:.0f}%).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="Nombre d'épisodes à jouer")
    args = parser.parse_args()
    main(args.n)
