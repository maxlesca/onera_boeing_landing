"""
analyze_domain_gap.py — Analyse de l'étude de domain gap (DOC §9.3, mission B2).
================================================================================
Consomme les detections.csv produits par yolo_runway_corners.py pour plusieurs
modèles, et produit :
  - domain_gap_detections.csv : toutes les détections consolidées (+ run, t)
  - domain_gap_summary.csv    : taux de détection par modèle × run × seuil,
                                confiance moyenne, 1re détection soutenue
  - domain_gap_detection_rate.png : taux de détection glissant vs temps de vol
  - domain_gap_confidence.png     : confiance glissante vs temps de vol

Usage :
    python analyze_domain_gap.py <dossier_etude> [--seuil 0.4] [--fenetre 15]

    <dossier_etude> contient un sous-dossier par modèle, chacun avec
    sortie_yolo/detections.csv (ex. resultats/domain_gap/pose_v8/...).
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# palette catégorielle validée (ordre FIXE — dataviz skill, mode clair)
COULEURS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"]
SURFACE, ENCRE, ENCRE_2, GRILLE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SEUILS = [0.25, 0.4, 0.6]


def charger(dossier: Path) -> pd.DataFrame:
    """Concatène les detections.csv de chaque modèle, extrait run et t du nom."""
    morceaux = []
    for sous in sorted(p for p in dossier.iterdir() if p.is_dir()):
        csv = sous / "sortie_yolo" / "detections.csv"
        if not csv.exists():
            print(f"(ignoré : {sous.name}, pas de detections.csv)")
            continue
        df = pd.read_csv(csv)
        df.insert(0, "model", sous.name)
        morceaux.append(df)
    tout = pd.concat(morceaux, ignore_index=True)
    infos = tout["fichier"].str.extract(r"ldg_sim_(?P<run>\d+)_idx_\d+_t_(?P<t>[\d.]+)\.jpg")
    tout["run"] = infos["run"].astype(int)
    tout["t"] = infos["t"].astype(float)
    return tout.sort_values(["model", "run", "t"]).reset_index(drop=True)


def premiere_detection_soutenue(serie_detecte: pd.Series, temps: pd.Series, n=3):
    """Premier t suivi d'au moins n échantillons consécutifs détectés."""
    consec = serie_detecte.rolling(n).sum()
    ou = np.where(consec >= n)[0]
    return float(temps.iloc[ou[0] - n + 1]) if len(ou) else np.nan


def resumer(tout: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for (modele, run), g in tout.groupby(["model", "run"]):
        for seuil in SEUILS:
            det = g["conf_coins"].fillna(0) >= seuil
            lignes.append({
                "model": modele, "run": run, "seuil": seuil,
                "images": len(g), "detections": int(det.sum()),
                "taux_pct": round(100 * det.mean(), 1),
                "conf_moyenne": round(g.loc[det, "conf_coins"].mean(), 3) if det.any() else np.nan,
                "t_premiere_detection_soutenue": premiere_detection_soutenue(
                    det.reset_index(drop=True), g["t"].reset_index(drop=True)),
            })
    return pd.DataFrame(lignes)


def tracer(tout: pd.DataFrame, sortie: Path, seuil: float, fenetre: int):
    modeles = list(tout["model"].unique())
    runs = sorted(tout["run"].unique())

    for quoi, fichier, titre, ylab in [
        ("taux", "domain_gap_detection_rate.png",
         f"Taux de détection glissant (fenêtre {fenetre} s, seuil {seuil}) — 1 image/s",
         "détection (%)"),
        ("conf", "domain_gap_confidence.png",
         f"Confiance glissante des détections (fenêtre {fenetre} s)", "confiance"),
    ]:
        fig, axes = plt.subplots(len(runs), 1, figsize=(10, 2.1 * len(runs)),
                                 sharex=False, facecolor=SURFACE)
        for ax, run in zip(np.atleast_1d(axes), runs):
            ax.set_facecolor(SURFACE)
            for i, modele in enumerate(modeles):
                g = tout[(tout["model"] == modele) & (tout["run"] == run)].sort_values("t")
                if quoi == "taux":
                    y = (g["conf_coins"].fillna(0) >= seuil).rolling(
                        fenetre, center=True, min_periods=5).mean() * 100
                else:
                    y = g["conf_coins"].rolling(fenetre, center=True, min_periods=3).mean()
                ax.plot(g["t"], y, lw=2, color=COULEURS[i % 4],
                        label=modele if run == runs[0] else None)
            if quoi == "taux":
                ax.set_ylim(-4, 104)
            else:
                ax.set_ylim(0, 1)
                for s, style in [(0.4, ":"), (0.6, "--")]:
                    ax.axhline(s, color=GRILLE, ls=style, lw=1, zorder=0)
            ax.set_ylabel(ylab, color=ENCRE_2, fontsize=9)
            ax.text(0.01, 0.86, f"run {run}", transform=ax.transAxes,
                    color=ENCRE, fontsize=10, fontweight="bold")
            ax.grid(color=GRILLE, lw=0.6)
            for cote in ("top", "right"):
                ax.spines[cote].set_visible(False)
            for cote in ("left", "bottom"):
                ax.spines[cote].set_color(GRILLE)
            ax.tick_params(colors=ENCRE_2, labelsize=8)
        np.atleast_1d(axes)[-1].set_xlabel("temps de vol t (s) — l'avion se rapproche du seuil",
                                           color=ENCRE_2, fontsize=9)
        fig.suptitle(titre, color=ENCRE, fontsize=11, fontweight="bold")
        fig.legend(loc="upper right", fontsize=8, frameon=False,
                   bbox_to_anchor=(0.99, 0.985), labelcolor=ENCRE_2)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(sortie / fichier, dpi=150, facecolor=SURFACE)
        print("figure ->", sortie / fichier)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dossier", help="dossier de l'étude (un sous-dossier par modèle)")
    ap.add_argument("--seuil", type=float, default=0.4, help="seuil pour les courbes de taux")
    ap.add_argument("--fenetre", type=int, default=15, help="fenêtre glissante (échantillons)")
    a = ap.parse_args()
    dossier = Path(a.dossier)

    tout = charger(dossier)
    tout.to_csv(dossier / "domain_gap_detections.csv", index=False)

    resume = resumer(tout)
    resume.to_csv(dossier / "domain_gap_summary.csv", index=False)
    print("\n=== TAUX DE DÉTECTION (%) par modèle × seuil (tous runs) ===")
    global_ = tout.assign(**{f"s{int(s*100)}": (tout["conf_coins"].fillna(0) >= s)
                             for s in SEUILS})
    print(global_.groupby("model")[[f"s{int(s*100)}" for s in SEUILS]]
          .mean().mul(100).round(1).to_string())
    print("\n=== 1re détection soutenue (s) par run, seuil 0.4 ===")
    print(resume[resume.seuil == 0.4].pivot(index="run", columns="model",
          values="t_premiere_detection_soutenue").round(0).to_string())

    tracer(tout, dossier, a.seuil, a.fenetre)


if __name__ == "__main__":
    main()
