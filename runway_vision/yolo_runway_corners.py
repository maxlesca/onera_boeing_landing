"""
yolo_coins_piste.py — Trouver les 4 coins de la piste dans les images.
======================================================================
Implémente l'architecture en étages du papier LARD 2.0 (DOC §9.8) :

  ÉTAGE 1 (optionnel) : un modèle de DÉTECTION calcule la boîte de la piste
                        sur l'image entière (réduite à --imgsz).
  ÉTAGE 2             : la boîte est DÉCOUPÉE dans l'image pleine résolution
                        (+ marge), et un modèle de COINS (pose/keypoints ou
                        segmentation) travaille sur ce gros plan — la piste
                        remplit le cadre quelle que soit la distance.
                        Les coins sont ensuite RECONVERTIS dans le repère de
                        l'image entière (offset du crop).

Modes :
  1 étage  : --coins <modele>                    (seg ou pose sur l'image entière
                                                  — le pipeline du papier EUCASS)
  2 étages : --detect <modele> --coins <modele>  (l'architecture LARD 2.0/Daedalean)

Modèles LARD (https://github.com/deel-ai-papers/Yolo_models_LARD_V2, via git lfs) :
  étage 1 : yolo_v8_models/yolov8detect_IN_ODD_best.pt   (ou une seg : sa boîte sert)
  étage 2 : yolo_v8_models/yolov8pose_best.pt            (keypoints = les coins)
  1 étage : yolo_v11_models/LARD_03_FULL_segN_Color_s1024_..._best.pt

Usage :
    python yolo_coins_piste.py <dossier_images> --coins LARD_03_FULL_segN...pt --crop 47,42
    python yolo_coins_piste.py <dossier_images> --detect yolov8detect_IN_ODD_best.pt \\
                               --coins yolov8pose_best.pt --crop 47,42

Sorties (sortie_yolo/) : detections.csv (coins étiquetés HG/HD/BD/BG, confiances)
                         + annotees/*.jpg (contrôle visuel : boîte + coins)
"""

import argparse
import csv
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------------------------------------- extraction coins
def quatre_coins_depuis_masque(polygone):
    """Contour de masque -> 4 sommets (approximation polygonale, EUCASS §9.7.4)."""
    contour = polygone.astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(contour, True)
    for frac in np.linspace(0.01, 0.15, 30):
        approx = cv2.approxPolyDP(contour, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return cv2.boxPoints(cv2.minAreaRect(contour))       # repli : rectangle orienté


def extraire_coins(res):
    """Coins depuis un résultat YOLO : keypoints (pose) sinon masque (seg).
    Retourne (coins (4,2) ou None, confiance)."""
    if res.boxes is None or len(res.boxes) == 0:
        return None, None
    i = int(res.boxes.conf.argmax())
    conf = float(res.boxes.conf[i])
    kp = getattr(res, "keypoints", None)
    if kp is not None and kp.xy is not None and kp.xy.shape[1] >= 4:
        return kp.xy[i][:4].cpu().numpy(), conf          # modèle POSE : direct
    if res.masks is not None and len(res.masks.xy) > i:
        return quatre_coins_depuis_masque(res.masks.xy[i]), conf   # modèle SEG
    return None, conf                                    # détection pure : boîte sans coins


def etiqueter(coins, largeur, hauteur):
    """Assigne les 4 coins aux labels {HG, HD, BD, BG} (coût minimal, EUCASS)."""
    refs = np.array([[0, 0], [largeur, 0], [largeur, hauteur], [0, hauteur]], np.float32)
    labels = ["haut_gauche", "haut_droit", "bas_droit", "bas_gauche"]
    meilleur, cout_min = None, np.inf
    for perm in permutations(range(4)):
        cout = sum(np.linalg.norm(coins[p] - refs[i]) for i, p in enumerate(perm))
        if cout < cout_min:
            cout_min, meilleur = cout, perm
    return {labels[i]: tuple(np.round(coins[p], 1)) for i, p in enumerate(meilleur)}


# ------------------------------------------------------------------- pipeline
def traiter_image(img, m_detect, m_coins, conf, imgsz, imgsz_coins, marge):
    """Retourne (coins (4,2) dans le repère IMAGE ENTIÈRE, conf_boite, conf_coins, boite)."""
    H, W = img.shape[:2]

    if m_detect is None:
        # ----- 1 ÉTAGE : le modèle de coins voit l'image entière
        res = m_coins.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
        coins, c = extraire_coins(res)
        return coins, None, c, None

    # ----- 2 ÉTAGES (LARD 2.0) — étage 1 : la boîte
    res1 = m_detect.predict(img, conf=conf, imgsz=imgsz, verbose=False)[0]
    if res1.boxes is None or len(res1.boxes) == 0:
        return None, None, None, None
    i = int(res1.boxes.conf.argmax())
    conf_boite = float(res1.boxes.conf[i])
    x1, y1, x2, y2 = res1.boxes.xyxy[i].cpu().numpy()

    # marge autour de la boîte, bornée à l'image (le crop se fait dans l'image
    # PLEINE RÉSOLUTION : c'est tout l'intérêt — la piste remplit le cadre)
    mx, my = (x2 - x1) * marge, (y2 - y1) * marge
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(W, int(x2 + mx)), min(H, int(y2 + my))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, conf_boite, None, None

    # étage 2 : les coins sur le gros plan
    res2 = m_coins.predict(crop, conf=conf, imgsz=imgsz_coins, verbose=False)[0]
    coins, conf_coins = extraire_coins(res2)
    if coins is None:
        return None, conf_boite, conf_coins, (x1, y1, x2, y2)

    coins = coins + np.array([x1, y1], np.float32)   # ⚠️ repère crop -> repère image
    return coins, conf_boite, conf_coins, (x1, y1, x2, y2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dossier", help="dossier d'images (déjà rognées, ou utiliser --crop)")
    ap.add_argument("--coins", default="yolo11n-seg.pt",
                    help="modèle de coins : pose (keypoints) ou segmentation")
    ap.add_argument("--detect", default=None,
                    help="modèle de détection étage 1 (active le mode 2 étages)")
    ap.add_argument("--conf", type=float, default=0.6, help="seuil de confiance (EUCASS : 0.6)")
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="taille d'entrée étage 1 / mode 1 étage (modèles LARD : s1024)")
    ap.add_argument("--imgsz-coins", type=int, default=640, help="taille d'entrée étage 2 (crop)")
    ap.add_argument("--marge", type=float, default=0.2, help="marge autour de la boîte (0.2 = 20 %%)")
    ap.add_argument("--crop", default=None,
                    help="'haut,bas' : overlays Windows à rogner AVANT détection (ex. 47,42)")
    ap.add_argument("--n", type=int, default=20, help="traiter 1 image sur n")
    args = ap.parse_args()

    crop_h, crop_b = (int(x) for x in args.crop.split(",")) if args.crop else (0, 0)
    images = sorted(Path(args.dossier).rglob("*.jpg"))[::args.n]
    m_coins = YOLO(args.coins)
    m_detect = YOLO(args.detect) if args.detect else None
    print(f"{len(images)} images | mode : {'2 étages (LARD 2.0)' if m_detect else '1 étage (EUCASS)'}")

    sortie = Path("sortie_yolo")
    (sortie / "annotees").mkdir(parents=True, exist_ok=True)
    (sortie / "non_detectees").mkdir(parents=True, exist_ok=True)   # l'atlas des échecs
    lignes = []

    for chemin in images:
        img = cv2.imread(str(chemin))
        if img is None:
            continue
        if crop_h or crop_b:
            img = img[crop_h: img.shape[0] - crop_b]   # ⚠️ c_y change (DOC §10.6-B5)

        coins, cb, cc, boite = traiter_image(img, m_detect, m_coins, args.conf,
                                             args.imgsz, args.imgsz_coins, args.marge)
        ligne = {"fichier": chemin.name, "conf_boite": cb, "conf_coins": cc}
        if coins is not None:
            etiq = etiqueter(coins, img.shape[1], img.shape[0])
            ligne.update({k: f"{u:.0f};{v:.0f}" for k, (u, v) in etiq.items()})
            if boite:
                cv2.rectangle(img, boite[:2], boite[2:], (255, 160, 0), 2)
            cv2.polylines(img, [coins.astype(int)], True, (0, 255, 0), 2)
            for nom, (u, v) in etiq.items():
                cv2.circle(img, (int(u), int(v)), 6, (0, 0, 255), -1)
                cv2.putText(img, nom[:2].upper(), (int(u) + 8, int(v)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(sortie / "annotees" / chemin.name), img)
        else:
            # image SANS détection : exportée telle quelle avec un bandeau,
            # pour voir OÙ le modèle échoue (piste trop lointaine ? occultée ?)
            cv2.putText(img, "NON DETECTEE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imwrite(str(sortie / "non_detectees" / chemin.name), img)
        lignes.append(ligne)
        print(f"  {chemin.name}: boite={cb} coins={cc}")

    champs = ["fichier", "conf_boite", "conf_coins",
              "haut_gauche", "haut_droit", "bas_droit", "bas_gauche"]
    with open(sortie / "detections.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=champs)
        w.writeheader()
        w.writerows(lignes)
    n_ok = sum(1 for l in lignes if l.get("haut_gauche"))
    print(f"\nCoins trouvés : {n_ok}/{len(lignes)} images -> {sortie.resolve()}")


if __name__ == "__main__":
    main()
