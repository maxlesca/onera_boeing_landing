"""
csv_vers_npz.py — Convertit le CSV d'états en dataset .npz prêt-à-entraîner.
============================================================================
Transforme `ldg_dataset_images_Maxime.csv` (la source de vérité, lisible) en
artefact d'entraînement binaire (DOC §7.7.1) : X (entrées rôle ①), Y (labels
rôle ②), + run et temps pour reconstruire les séquences du CfC.

Décisions encodées ici (DOC §8.4 et §10.1) :
  - ENTRÉES ① = attitude, vitesses angulaires, vitesses corps et NED,
    déviations ILS (en mètres). PAS la position vraie ni le vent (rôle ③).
  - LABELS  ② = TOUTES les commandes : longitudinal (profondeur), lateral
    (ailerons), directional (palonnier — constant 0 dans ce dataset), stabilizer
    (trim), throttle_left (= right partout, une seule colonne).
  - Les lignes sans état (NaN) sont écartées et comptées.
  - ⚠️ Découpage train/val PAR RUN, jamais par frames aléatoires : deux frames
    consécutives sont quasi identiques -> les mélanger mettrait "presque le
    jeu de test" dans le train (fuite de données).
  - Les bornes min/max de normalisation sont calculées sur le TRAIN SEUL et
    sauvées dans le npz (contrat n°2 du DOC §7.7.1).

Usage :
    python csv_vers_npz.py <dataset_Maxime.zip ou chemin.csv> [--val-runs 8] [--out datasets]
"""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ENTREES = ["pitch", "bank", "heading", "p", "q", "r", "u", "v", "w",
           "northsouth_velocity", "eastwest_velocity", "vertical_velocity",
           "localizer_error_m", "glideslope_error_m"]
COLONNES_ILS = ["localizer_error_m", "glideslope_error_m"]   # retirées par --sans-ils
# TOUTES les commandes enregistrées (choix : ne rien exclure).
# NB dataset actuel : directional constant = 0 (le réseau apprendra « 0 » —
# sans coût, et le pipeline est prêt pour un futur dataset avec vent de
# travers) ; stabilizer = trim 0/1 ; throttle_left == throttle_right (une
# seule colonne suffit).
LABELS = ["longitudinal", "lateral", "directional", "stabilizer", "throttle_left"]


def charger_csv(chemin: Path) -> pd.DataFrame:
    if chemin.suffix.lower() == ".zip":
        with zipfile.ZipFile(chemin) as z:
            nom = next(n for n in z.namelist() if n.endswith(".csv"))
            print(f"Lecture de {nom} dans le zip...")
            return pd.read_csv(io.BytesIO(z.read(nom)), sep=";")
    return pd.read_csv(chemin, sep=";")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="le .zip du dataset ou directement le .csv")
    ap.add_argument("--val-runs", default="8",
                    help="runs réservés à la validation, séparés par des virgules (défaut : 8)")
    ap.add_argument("--out", default="datasets", help="dossier de sortie")
    ap.add_argument("--sans-ils", action="store_true",
                    help="retire les déviations ILS des entrées (pipeline YOLO→CfC, DOC §9.8 : "
                         "la vision remplacera ces colonnes)")
    args = ap.parse_args()

    global ENTREES
    if args.sans_ils:
        ENTREES = [c for c in ENTREES if c not in COLONNES_ILS]
        print(f"Mode --sans-ils : entrées réduites à {len(ENTREES)} colonnes (ILS exclu)")

    df = charger_csv(Path(args.source))
    n_total = len(df)
    df = df.dropna(subset=["image_filename"] + ENTREES + LABELS).copy()
    print(f"{n_total} lignes lues, {n_total - len(df)} écartées (NaN), {len(df)} conservées")

    df["run"] = df["simulationindex"].astype(int)
    df = df.sort_values(["run", "time"]).reset_index(drop=True)

    val_runs = {int(r) for r in args.val_runs.split(",")}
    runs_presents = set(df["run"].unique())
    if not val_runs & runs_presents:
        raise SystemExit(f"Aucun des runs de validation {val_runs} n'existe (runs : {sorted(runs_presents)})")
    masque_val = df["run"].isin(val_runs)

    sortie = Path(args.out)
    sortie.mkdir(parents=True, exist_ok=True)

    # bornes de normalisation : SUR LE TRAIN SEUL (contrat DOC §7.7.1 n°2)
    train = df[~masque_val]
    bornes = {
        "entrees": ENTREES, "labels": LABELS,
        "x_min": train[ENTREES].min().tolist(), "x_max": train[ENTREES].max().tolist(),
        "y_min": train[LABELS].min().tolist(), "y_max": train[LABELS].max().tolist(),
    }

    for nom, part in [("train", train), ("val", df[masque_val])]:
        np.savez_compressed(
            sortie / f"landing_{nom}.npz",
            X=part[ENTREES].to_numpy(np.float32),         # (N, 14) entrées BRUTES
            Y=part[LABELS].to_numpy(np.float32),          # (N, 3)  labels BRUTS
            run=part["run"].to_numpy(np.int32),           # pour reconstruire les séquences
            t=part["time"].to_numpy(np.float32),
            image=part["image_filename"].to_numpy(),      # lien vers l'image (usage vision)
            x_min=np.array(bornes["x_min"], np.float32),  # bornes embarquées dans le fichier
            x_max=np.array(bornes["x_max"], np.float32),
            y_min=np.array(bornes["y_min"], np.float32),
            y_max=np.array(bornes["y_max"], np.float32),
        )
        runs_ici = sorted(part["run"].unique())
        print(f"  {nom}: {len(part):6d} paires, runs {runs_ici} "
              f"-> {sortie / f'landing_{nom}.npz'}")

    (sortie / "normalization_bounds.json").write_text(
        json.dumps(bornes, indent=2), encoding="utf-8")
    print("Bornes de normalisation (train seul) -> normalization_bounds.json")
    print("\nRelecture type :  d = np.load('landing_train.npz', allow_pickle=True)")
    print("                  X_norm = (d['X'] - d['x_min']) / (d['x_max'] - d['x_min'] + 1e-10)")


if __name__ == "__main__":
    main()
