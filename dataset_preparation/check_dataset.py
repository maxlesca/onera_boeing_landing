"""
check_dataset.py — Contrôle qualité automatique du dataset d'images MSFS.
=========================================================================
Étape 1 de la démarche (DOC §10.7). Vérifie la STRUCTURE (noms de fichiers,
cadence, index) sur TOUTES les images, et la QUALITÉ PIXEL (corruption,
frames gelées, noires, floues, overlays) sur un échantillon.

Usage :
    python check_dataset.py <dossier_images OU archive.zip> [--sample 25] [--out rapport]

    Accepte un dossier d'images OU directement un .zip (lu en streaming, sans
    extraction — pratique quand le disque est plein ou le zip volumineux).

    --sample N : ne fait les contrôles pixel que sur 1 image sur N (défaut 25).
                 Mettre 1 pour tout contrôler (long : ~1 h pour 10 000 images).
    --out DIR  : dossier de sortie du rapport (défaut : ./rapport_dataset)

Sorties :
    rapport_dataset/resume.txt        — le bilan lisible
    rapport_dataset/anomalies.csv     — une ligne par anomalie détectée
    rapport_dataset/frames.csv        — les métriques par frame échantillonnée
"""

import argparse
import collections
import csv
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV manquant : pip install opencv-python")


class Source:
    """Accès uniforme aux images, qu'elles soient dans un dossier ou un .zip."""

    def __init__(self, chemin: Path):
        self.zip = zipfile.ZipFile(chemin) if chemin.suffix.lower() == ".zip" else None
        self.racine = chemin

    def noms(self):
        if self.zip:
            return [n for n in self.zip.namelist() if not n.endswith("/")]
        return [str(p.relative_to(self.racine)) for p in self.racine.rglob("*") if p.is_file()]

    def lire_image(self, nom):
        """Retourne l'image BGR (ou None si illisible)."""
        if self.zip:
            data = np.frombuffer(self.zip.read(nom), np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        return cv2.imread(str(self.racine / nom))

# ---------------------------------------------------------------- paramètres
PATTERN = re.compile(r"ldg_sim_(\d+)_idx_(\d+)_t_([\d.]+)\.(jpg|jpeg|png)$")
DT_ATTENDU = 0.04          # cadence attendue (25 Hz) — adapter si besoin
SEUIL_NOIR = 10.0          # moyenne de pixels sous laquelle l'image est "noire"
SEUIL_BLANC = 245.0        # ... au-dessus de laquelle elle est "blanche"
SEUIL_FLOU = 15.0          # variance du Laplacien sous laquelle c'est "flou"
SEUIL_GELE = 0.5           # diff. moyenne (0-255) entre frames consécutives
BANDE_HAUT = 45            # hauteur (px) de la bande d'overlay suspectée en haut
BANDE_BAS = 40             # ... en bas (barre des tâches)


def parser_fichiers(source: Source):
    """Retourne {run: [(idx, t, nom), ...]} et la liste des noms invalides."""
    runs, invalides = collections.defaultdict(list), []
    for nom in sorted(source.noms()):
        if not nom.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        m = PATTERN.search(Path(nom).name)
        if not m:
            invalides.append(nom)
            continue
        runs[int(m.group(1))].append((int(m.group(2)), float(m.group(3)), nom))
    for r in runs:
        runs[r].sort()
    return runs, invalides


def controle_structure(runs, anomalies):
    """Contrôles A (DOC §10.5) : cadence, index, cohérence des noms."""
    lignes = []
    tous_idx = []
    for run, frames in sorted(runs.items()):
        idx = [f[0] for f in frames]
        ts = [f[1] for f in frames]
        tous_idx += idx

        manquants = set(range(idx[0], idx[-1] + 1)) - set(idx)
        doublons = [i for i, c in collections.Counter(idx).items() if c > 1]
        dts = np.round(np.diff(ts), 4)
        dt_irreguliers = int(np.sum(np.abs(dts - DT_ATTENDU) > 1e-3))

        if manquants:
            anomalies.append(("structure", f"run {run}",
                              f"{len(manquants)} index manquants (ex: {sorted(manquants)[:5]})"))
        if doublons:
            anomalies.append(("structure", f"run {run}", f"index dupliqués : {doublons[:5]}"))
        if dt_irreguliers:
            anomalies.append(("structure", f"run {run}",
                              f"{dt_irreguliers} intervalles != {DT_ATTENDU}s"))
        if ts[0] > 1e-6:
            anomalies.append(("structure", f"run {run}",
                              f"premier t = {ts[0]:.3f}s au lieu de 0.000 "
                              f"(~{round(ts[0] / DT_ATTENDU)} frames absentes au début ?)"))

        lignes.append(f"  run {run:4d} : {len(frames):5d} frames, "
                      f"t = {ts[0]:8.3f} -> {ts[-1]:8.3f} s, "
                      f"dt irréguliers = {dt_irreguliers}, "
                      f"manquants = {len(manquants)}")

    # continuité de l'index GLOBAL entre runs
    tous_idx.sort()
    trous_globaux = set(range(tous_idx[0], tous_idx[-1] + 1)) - set(tous_idx)
    if trous_globaux:
        anomalies.append(("structure", "global",
                          f"{len(trous_globaux)} index absents de la numérotation globale"))
    return lignes


def metriques_image(img_gris):
    """Métriques pixel d'une image en niveaux de gris."""
    return {
        "moyenne": float(img_gris.mean()),
        "ecart_type": float(img_gris.std()),
        "flou_laplacien": float(cv2.Laplacian(img_gris, cv2.CV_64F).var()),
    }


def controle_pixels(source: Source, runs, pas, anomalies, dossier_sortie: Path):
    """Contrôles B (DOC §10.5) sur 1 frame sur `pas` + détection d'overlays."""
    lignes_csv = []
    tailles = collections.Counter()
    bandes_haut, bandes_bas, bandes_milieu = [], [], []
    n_echantillon = 0

    for run, frames in sorted(runs.items()):
        gris_prec, chemin_prec = None, None
        for k in range(0, len(frames), pas):
            idx, t, nom = frames[k]
            chemin = Path(nom)
            img = source.lire_image(nom)
            if img is None:
                anomalies.append(("pixel", chemin.name, "image illisible/corrompue"))
                continue
            n_echantillon += 1
            tailles[img.shape[:2]] += 1
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            m = metriques_image(gris)
            m.update(run=run, idx=idx, t=t, fichier=chemin.name)

            if m["moyenne"] < SEUIL_NOIR:
                anomalies.append(("pixel", chemin.name, f"image quasi noire (moy={m['moyenne']:.0f})"))
            if m["moyenne"] > SEUIL_BLANC:
                anomalies.append(("pixel", chemin.name, f"image quasi blanche (moy={m['moyenne']:.0f})"))
            if m["flou_laplacien"] < SEUIL_FLOU:
                anomalies.append(("pixel", chemin.name,
                                  f"flou suspect (laplacien={m['flou_laplacien']:.1f})"))

            # frame gelée ? (comparaison avec la frame échantillonnée précédente
            # du même run, réduite pour aller vite)
            petit = cv2.resize(gris, (160, 90))
            if gris_prec is not None:
                diff = float(np.abs(petit.astype(np.float32) - gris_prec).mean())
                m["diff_frame_prec"] = diff
                if diff < SEUIL_GELE:
                    anomalies.append(("pixel", chemin.name,
                                      f"quasi identique à {chemin_prec} (diff={diff:.2f}) "
                                      "— sim en pause pendant la capture ?"))
            gris_prec, chemin_prec = petit, chemin.name

            # accumulateurs pour la détection d'overlays statiques
            bandes_haut.append(cv2.resize(gris[:BANDE_HAUT], (320, 8)).astype(np.float32))
            bandes_bas.append(cv2.resize(gris[-BANDE_BAS:], (320, 8)).astype(np.float32))
            h = gris.shape[0]
            bandes_milieu.append(cv2.resize(gris[h // 2 - 20:h // 2 + 20], (320, 8)).astype(np.float32))

            lignes_csv.append(m)

    # --- détection d'overlays : une zone STATIQUE d'une frame à l'autre alors
    # que la scène bouge = un élément d'interface incrusté (DOC §10.6, C2).
    verdict_overlay = "non testé (échantillon trop petit)"
    if len(bandes_haut) > 10:
        std_haut = float(np.std(np.stack(bandes_haut), axis=0).mean())
        std_bas = float(np.std(np.stack(bandes_bas), axis=0).mean())
        std_mil = float(np.std(np.stack(bandes_milieu), axis=0).mean())
        verdict_overlay = (f"variabilité temporelle : haut={std_haut:.1f}, "
                           f"milieu={std_mil:.1f}, bas={std_bas:.1f}")
        # la bande est "statique" si elle varie beaucoup moins que le milieu
        if std_haut < 0.5 * std_mil:
            anomalies.append(("overlay", f"bande haute ({BANDE_HAUT}px)",
                              "zone quasi statique dans le temps -> overlay probable "
                              "(barre de titre/menu ?) -> à rogner + recalculer c_y"))
        if std_bas < 0.5 * std_mil:
            anomalies.append(("overlay", f"bande basse ({BANDE_BAS}px)",
                              "zone quasi statique dans le temps -> overlay probable "
                              "(barre des tâches ?) -> à rogner + recalculer c_y"))

    if len(tailles) > 1:
        anomalies.append(("pixel", "global", f"plusieurs résolutions présentes : {dict(tailles)}"))

    # écrire le CSV par frame
    if lignes_csv:
        champs = sorted({c for l in lignes_csv for c in l})
        with open(dossier_sortie / "frames.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=champs)
            w.writeheader()
            w.writerows(lignes_csv)

    return n_echantillon, tailles, verdict_overlay


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dossier", help="dossier contenant les images du dataset")
    ap.add_argument("--sample", type=int, default=25,
                    help="contrôles pixel sur 1 frame sur N (défaut 25, 1 = toutes)")
    ap.add_argument("--out", default="rapport_dataset", help="dossier du rapport")
    args = ap.parse_args()

    dossier = Path(args.dossier)
    sortie = Path(args.out)
    sortie.mkdir(parents=True, exist_ok=True)
    anomalies = []

    print(f"Analyse de {dossier} ...")
    source = Source(dossier)
    runs, invalides = parser_fichiers(source)
    if not runs:
        sys.exit("Aucune image au format attendu (ldg_sim_*_idx_*_t_*.jpg) trouvée.")
    for nom in invalides:
        anomalies.append(("nommage", nom, "nom de fichier hors schéma"))

    lignes_runs = controle_structure(runs, anomalies)
    n_ech, tailles, verdict_overlay = controle_pixels(source, runs, args.sample, anomalies, sortie)

    # ------------------------------------------------------------- rapport
    total = sum(len(f) for f in runs.values())
    resume = [
        "=" * 72,
        "RAPPORT QUALITÉ DATASET",
        "=" * 72,
        f"Dossier          : {dossier}",
        f"Images valides   : {total}  (noms invalides : {len(invalides)})",
        f"Runs             : {sorted(runs)}",
        f"Cadence attendue : {DT_ATTENDU}s ({1/DT_ATTENDU:.0f} Hz)",
        "",
        "Par run :",
        *lignes_runs,
        "",
        f"Contrôles pixel  : {n_ech} frames échantillonnées (1/{args.sample})",
        f"Résolutions      : {dict(tailles)}",
        f"Test overlays    : {verdict_overlay}",
        "",
        f"ANOMALIES : {len(anomalies)}",
        "-" * 72,
        *(f"  [{fam:9s}] {ou:32s} {msg}" for fam, ou, msg in anomalies[:60]),
        *(["  ... (voir anomalies.csv pour la liste complète)"] if len(anomalies) > 60 else []),
    ]
    texte = "\n".join(resume)
    print(texte)
    (sortie / "resume.txt").write_text(texte, encoding="utf-8")
    with open(sortie / "anomalies.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["famille", "localisation", "description"])
        w.writerows(anomalies)
    print(f"\nRapport écrit dans : {sortie.resolve()}")


if __name__ == "__main__":
    main()
