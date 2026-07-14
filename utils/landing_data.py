# -*- coding: utf-8 -*-
"""
File: landing_data.py  (nouveau module — n'existe pas chez Tudor)

Purpose:
    Équivalent de `get_data` (utils/data.py) pour le problème ATTERRISSAGE.
    Assemble :
      - le npz produit par dataset_preparation/csv_to_npz.py --sans-ils
        (X capteurs BRUTS, Y gouvernes BRUTES, run, t, image, et les BORNES de
        normalisation calculées sur le train seul — la source de vérité) ;
      - le detections.csv du grand run YOLO (yolo_runway_corners.py --n 1) ;
    en tableaux (fenêtres, features, seq_len) normalisés [0,1] — le format
    exact que consomment DatasetController et Lightning_Model de Tudor.

    L'entraînement ne touche JAMAIS aux images : la vision a été pré-calculée
    par YOLO (gelé) au grand run (DOC §9.8), et les états viennent du npz.

    Différence structurelle avec le drone : les runs n'ont pas tous la même
    longueur -> chaque run est découpé en FENÊTRES de seq_len pas (stride
    réglable), jamais à cheval sur deux runs ; chaque fenêtre devient une
    « trajectoire » au sens de Tudor.

Décisions encodées (DOC §9.8 / §8.4) :
    - vision = 4 coins BRUTS normalisés par (largeur, hauteur) image
      + confiance + drapeau de validité. Pas de détection -> validité 0 et
      coins GELÉS à la dernière valeur (0.5 = centre avant toute détection).
    - capteurs/labels : normalisés avec LES BORNES EMBARQUÉES DANS LE NPZ
      (contrat n°2, DOC §7.7.1 — aucune table dupliquée ici).
    - le découpage train/val par run est déjà fait DANS les npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CORNER_NAMES = ["haut_gauche", "haut_droit", "bas_droit", "bas_gauche"]
# les 8+2 canaux vision, dans un ORDRE FIXE (contrat n°1, DOC §7.7.1),
# insérés AVANT les canaux capteurs du npz
VISION_LABELS = [f"{c}_{ax}" for c in CORNER_NAMES for ax in ("u", "v")] + ["conf", "valid"]


def _parse_corners(detections_csv: str | Path, largeur: int, hauteur: int) -> pd.DataFrame:
    """detections.csv -> colonnes coin_u/coin_v ∈ [0,1] + conf (NaN si non détecté)."""
    det = pd.read_csv(detections_csv)
    out = pd.DataFrame({"image": det["fichier"]})
    for c in CORNER_NAMES:
        uv = det[c].str.split(";", expand=True).astype(float)
        out[f"{c}_u"] = uv[0] / largeur
        out[f"{c}_v"] = uv[1] / hauteur
    out["conf"] = det["conf_coins"]
    return out


def _vision_matrix(df: pd.DataFrame) -> np.ndarray:
    """Politique 'pas de détection' (DOC §9.8) : validité 0, coins gelés à la
    dernière valeur vue (0.5 avant la première), confiance 0. -> (10, T)."""
    coins = [f"{c}_{ax}" for c in CORNER_NAMES for ax in ("u", "v")]
    valid = df[coins[0]].notna().astype(np.float32)
    coins_ffill = df[coins].ffill().fillna(0.5)
    conf = df["conf"].fillna(0.0)
    vision = pd.concat([coins_ffill, conf, valid], axis=1)
    return vision.to_numpy(np.float32).T


def get_landing_data(npz_path: str | Path,
                     detections_csv: str | Path,
                     seq_len: int = 64,
                     stride: int = 16,
                     image_size: tuple[int, int] = (1920, 991),
                     normalized: bool = True):
    """
    Retourne (input_array, output_array) au format Tudor :
    (n_fenetres, features, seq_len), features = [10 vision | capteurs du npz].

    npz_path : landing_train.npz OU landing_val.npz (datasets/no_ils/) —
    le choix du npz EST le choix du split.
    """
    d = np.load(npz_path, allow_pickle=True)
    x_min, x_max = d["x_min"][None, :, None], d["x_max"][None, :, None]
    y_min, y_max = d["y_min"][None, :, None], d["y_max"][None, :, None]

    etats = pd.DataFrame({"image": [Path(p).name for p in d["image"]],
                          "run": d["run"], "t": d["t"]})
    coins = _parse_corners(detections_csv, *image_size)
    joint = etats.merge(coins, on="image", how="left")
    n_sans_detection = int(joint["conf"].isna().sum())
    print(f"{Path(npz_path).name} : {len(joint)} frames, "
          f"{n_sans_detection} sans détection YOLO (validité=0, coins gelés)")

    fenetres_x, fenetres_y = [], []
    for run in np.unique(d["run"]):                       # jamais à cheval sur 2 runs
        masque = d["run"] == run
        ordre = np.argsort(d["t"][masque])
        x_capteurs = d["X"][masque][ordre].T              # (capteurs, T_run) BRUTS
        y = d["Y"][masque][ordre].T                       # (3, T_run) BRUTS
        vision = _vision_matrix(joint[masque].iloc[ordre])  # (10, T_run), déjà [0,1]

        if normalized:
            x_capteurs = (x_capteurs - x_min[0]) / (x_max[0] - x_min[0] + 1e-10)
            y = (y - y_min[0]) / (y_max[0] - y_min[0] + 1e-10)

        x = np.concatenate([vision, x_capteurs], axis=0)  # (10 + capteurs, T_run)
        for debut in range(0, x.shape[1] - seq_len + 1, stride):
            fenetres_x.append(x[:, debut:debut + seq_len])
            fenetres_y.append(y[:, debut:debut + seq_len])

    input_array = np.stack(fenetres_x).astype(np.float32)
    output_array = np.stack(fenetres_y).astype(np.float32)
    print(f"  -> {len(input_array)} fenêtres de {seq_len} pas, "
          f"{input_array.shape[1]} entrées, {output_array.shape[1]} sorties")
    return input_array, output_array


def denormalize_controls(y_norm: np.ndarray, npz_path: str | Path) -> np.ndarray:
    """[0,1] -> unités physiques des gouvernes, avec les bornes du npz
    (pour la boucle fermée et les tracés)."""
    d = np.load(npz_path, allow_pickle=True)
    return y_norm * (d["y_max"] - d["y_min"]) + d["y_min"]
