"""
extraire_rosbag.py — rosbag -> images .jpg + états .csv + contrôle de synchro.
==============================================================================
Étape 1 de la démarche (DOC §10.7). Squelette PRÊT À ADAPTER : les deux choses
à renseigner sont les NOMS DES TOPICS et les CHAMPS du message d'état — demande
à Tudor la liste exacte (`ros2 bag info mon.bag` la donne aussi).

Prérequis :  pip install rosbags opencv-python pandas
Usage     :  python extraire_rosbag.py <fichier.bag ou dossier ros2> --out extraction
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

# ------------------------------------------------------------- À ADAPTER ---
TOPIC_IMAGE = "/camera/image"          # <- le vrai nom : voir `ros2 bag info`
TOPIC_ETAT = "/aircraft/state"         # <- idem
CHAMPS_ETAT = ["lat", "lon", "alt", "phi", "theta", "psi",
               "vn", "ve", "vz"]       # <- les champs du message d'état
TOLERANCE_NS = 20_000_000              # 20 ms : tolérance d'appariement image<->état
# ----------------------------------------------------------------------------


def extraire(chemin_bag: Path, sortie: Path):
    (sortie / "images").mkdir(parents=True, exist_ok=True)
    t_images, lignes_etat = [], []

    with AnyReader([chemin_bag]) as reader:
        print("Topics disponibles dans ce bag :")
        for c in reader.connections:
            print(f"  {c.topic}  ({c.msgtype}, {c.msgcount} messages)")

        conns_img = [c for c in reader.connections if c.topic == TOPIC_IMAGE]
        conns_etat = [c for c in reader.connections if c.topic == TOPIC_ETAT]
        if not conns_img or not conns_etat:
            raise SystemExit("Topics non trouvés : adapte TOPIC_IMAGE / TOPIC_ETAT ci-dessus.")

        for conn, t_ns, raw in reader.messages(connections=conns_img):
            msg = reader.deserialize(raw, conn.msgtype)
            # ⚠️ header.stamp (horodatage à l'ACQUISITION) est préférable à t_ns
            # (horodatage à l'ENREGISTREMENT) : voir DOC §10.6-A2.
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
            if msg.encoding in ("rgb8",):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(sortie / "images" / f"{stamp}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            t_images.append(stamp)

        for conn, t_ns, raw in reader.messages(connections=conns_etat):
            m = reader.deserialize(raw, conn.msgtype)
            stamp = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            lignes_etat.append([stamp] + [getattr(m, ch) for ch in CHAMPS_ETAT])

    df_etat = pd.DataFrame(lignes_etat, columns=["t_ns"] + CHAMPS_ETAT).sort_values("t_ns")
    df_etat.to_csv(sortie / "etats.csv", index=False)

    # ---------- le contrôle de synchronisation (DOC §10.5-A5) ----------
    df_img = pd.DataFrame({"t_ns": sorted(t_images)})
    fusion = pd.merge_asof(df_img, df_etat, on="t_ns",
                           direction="nearest", tolerance=TOLERANCE_NS)
    orphelines = int(fusion[CHAMPS_ETAT[0]].isna().sum())
    fusion.to_csv(sortie / "images_avec_etats.csv", index=False)

    print(f"\n{len(t_images)} images, {len(df_etat)} états extraits -> {sortie.resolve()}")
    print(f"Images SANS état à moins de {TOLERANCE_NS/1e6:.0f} ms : {orphelines}"
          f"  ({'OK' if orphelines == 0 else 'PROBLÈME DE SYNCHRO -> DOC §10.6 famille A'})")
    dts = np.diff(sorted(t_images)) / 1e9
    print(f"Cadence images : {1/np.median(dts):.1f} Hz "
          f"(dt min {dts.min():.4f}s / max {dts.max():.4f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("--out", default="extraction")
    a = ap.parse_args()
    extraire(Path(a.bag), Path(a.out))
