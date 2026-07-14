"""
crop_images.py — Rogne les bordures Windows (barre de titre/menu MSFS en haut,
barre des tâches en bas) des images du dataset.
===================================================================================
Trois modes, cumulables :

  --auto        DÉTECTE la hauteur des bandes tout seul : sur un échantillon de
                frames, mesure la variabilité TEMPORELLE de chaque ligne de
                pixels ; une ligne qui ne change jamais alors que la scène
                bouge = un overlay (DOC §10.6-C2).
  --haut/--bas  crop fixe si tu connais déjà les hauteurs (ex. 47 et 42).
  --si-bordure  ne rogne QUE les images où la bordure est réellement présente :
                chaque image est comparée à une bande de référence (la médiane
                des bandes de l'échantillon) ; si sa bande du haut/bas ne
                ressemble pas à la référence, l'image est copiée intacte.
                → sans risque sur un dataset mixte (images déjà propres +
                captures avec UI mélangées).

Écrit `crop_info.json` : bornes appliquées, statistiques, et rappel de la
correction du centre optique (c_y' = c_y − haut ; DOC §10.6-B5) à reporter
dans CONVENTIONS.md.

⚠️ Espace disque : matérialiser 16 184 images rognées ≈ la taille du dataset
d'origine (ton C: est presque plein). Utilise --apercu ou --limite pour
valider, matérialise sur un disque avec de la place, ou rogne À LA VOLÉE dans
ton Dataset PyTorch (img[haut:H-bas]) — la matérialisation n'est indispensable
que pour Ultralytics/YOLO (qui lit des fichiers).

Usage :
    python crop_images.py <zip ou dossier> --auto --apercu
    python crop_images.py <zip ou dossier> --auto --si-bordure --out D:\\images_crop
    python crop_images.py <zip ou dossier> --haut 47 --bas 42 --limite 100
"""

import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

LARGEUR_REF = 320      # largeur de travail des bandes (réduites pour la vitesse)
SEUIL_PRESENCE = 4.0   # écart moyen (0-255) sous lequel une bande = la référence


def iter_images(source: Path):
    """Rend (nom, bytes) pour chaque jpg, que la source soit un zip ou un dossier."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            for n in z.namelist():
                if n.lower().endswith(".jpg"):
                    yield Path(n).name, z.read(n)
    else:
        for p in sorted(source.rglob("*.jpg")):
            yield p.name, p.read_bytes()


def decoder(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def bande(gris, haut=None, bas=None):
    """Extrait la bande du haut ou du bas, réduite à LARGEUR_REF de large."""
    zone = gris[:haut] if haut else gris[-bas:]
    return cv2.resize(zone, (LARGEUR_REF, zone.shape[0])).astype(np.float32)


def analyser_echantillon(source: Path, n_echantillon=60):
    """1) Détecte les hauteurs de bande (variabilité temporelle par ligne).
       2) Construit les bandes de RÉFÉRENCE (médiane) pour le mode --si-bordure."""
    print(f"Analyse d'un échantillon de {n_echantillon} frames...")
    tous = list(iter_images(source))
    pas = max(1, len(tous) // n_echantillon)
    frames = []
    for nom, data in tous[::pas][:n_echantillon]:
        img = decoder(data)
        if img is not None:
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gris, (LARGEUR_REF, gris.shape[0])).astype(np.float32))
    pile = np.stack(frames)                          # (N, H, LARGEUR_REF)
    std_ligne = pile.std(axis=0).mean(axis=1)        # variabilité temporelle par ligne
    h = len(std_ligne)
    ref = np.median(std_ligne[h // 3: 2 * h // 3])   # variabilité "normale" (centre image)
    statique = std_ligne < 0.25 * ref

    haut = 0
    while haut < h // 4 and statique[haut]:
        haut += 1
    bas = 0
    while bas < h // 4 and statique[h - 1 - bas]:
        bas += 1
    haut = haut + 2 if haut else 0                   # marge anti-repliement jpg
    bas = bas + 2 if bas else 0
    print(f"  variabilité centre ~{ref:.1f} ; bandes statiques : haut={haut}px, bas={bas}px")

    ref_haut = np.median(pile[:, :haut], axis=0) if haut else None
    ref_bas = np.median(pile[:, h - bas:], axis=0) if bas else None
    return haut, bas, ref_haut, ref_bas


def a_la_bordure(gris, ref_bande, haut=None, bas=None):
    """Cette image porte-t-elle la bordure ? Comparaison à la bande de référence."""
    if ref_bande is None:
        return False
    b = bande(gris, haut=haut, bas=bas)
    if b.shape != ref_bande.shape:
        b = cv2.resize(b, (ref_bande.shape[1], ref_bande.shape[0]))
    return float(np.abs(b - ref_bande).mean()) < SEUIL_PRESENCE


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="le .zip du dataset ou un dossier d'images")
    ap.add_argument("--auto", action="store_true", help="détecter les hauteurs automatiquement")
    ap.add_argument("--haut", type=int, default=0)
    ap.add_argument("--bas", type=int, default=0)
    ap.add_argument("--si-bordure", action="store_true",
                    help="ne rogner que les images où la bordure est détectée (les autres sont copiées intactes)")
    ap.add_argument("--out", default="images_crop", help="dossier de sortie")
    ap.add_argument("--apercu", action="store_true", help="ne traiter que 5 images")
    ap.add_argument("--limite", type=int, default=0, help="ne traiter que N images")
    args = ap.parse_args()

    source = Path(args.source)
    ref_haut = ref_bas = None
    if args.auto or args.si_bordure:
        haut, bas, ref_haut, ref_bas = analyser_echantillon(source)
        if not args.auto:                            # hauteurs imposées, mais refs quand même
            haut, bas = args.haut, args.bas
    else:
        haut, bas = args.haut, args.bas
    if haut == 0 and bas == 0:
        raise SystemExit("Rien à rogner (donne --haut/--bas ou utilise --auto).")

    # Garde-fou : on n'écrit JAMAIS dans la source (le dataset d'origine reste intact).
    sortie = Path(args.out).resolve()
    src = source.resolve()
    if sortie == src or (src.is_dir() and src in sortie.parents) or sortie == src.parent / src.stem:
        raise SystemExit("Refus : --out pointe vers le dataset d'origine. "
                         "Choisis un dossier de sortie séparé (les originaux ne sont jamais modifiés).")
    sortie.mkdir(parents=True, exist_ok=True)
    limite = 5 if args.apercu else (args.limite or None)

    n_total = n_rognees = n_intactes = 0
    resolution = None
    for nom, data in iter_images(source):
        img = decoder(data)
        if img is None:
            continue
        h = img.shape[0]
        if args.si_bordure:
            gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            c_haut = haut if a_la_bordure(gris, ref_haut, haut=haut) else 0
            c_bas = bas if a_la_bordure(gris, ref_bas, bas=bas) else 0
        else:
            c_haut, c_bas = haut, bas

        if c_haut or c_bas:
            img = img[c_haut: h - c_bas]
            n_rognees += 1
        else:
            n_intactes += 1
        cv2.imwrite(str(sortie / nom), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        resolution = (img.shape[1], img.shape[0])
        n_total += 1
        if limite and n_total >= limite:
            break

    info = {
        "mode": "si-bordure" if args.si_bordure else "systematique",
        "crop_haut_px": haut, "crop_bas_px": bas,
        "images_traitees": n_total, "rognees": n_rognees, "laissees_intactes": n_intactes,
        "resolution_apres_crop": resolution,
        "correction_centre_optique": f"c_y' = c_y - {haut} sur les images rognées (DOC §10.6-B5)",
        "qualite_jpeg": 95,
    }
    (sortie / "crop_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{n_total} images traitées -> {sortie.resolve()}")
    print(f"  rognées : {n_rognees} | laissées intactes : {n_intactes}")
    print(f"crop_info.json écrit. ATTENTION : c_y diminue de {haut} px sur les images rognées.")


if __name__ == "__main__":
    main()
