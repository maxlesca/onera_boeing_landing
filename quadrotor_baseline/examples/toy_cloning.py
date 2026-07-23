"""
toy_cloning.py — TON premier behavioral cloning, sur un problème jouet 1D.
==========================================================================
L'exercice du DOC §7.6, complet et exécutable (CPU, ~1 minute).

Le problème : un chariot 1D (position, vitesse) doit rejoindre l'origine.
  - L'EXPERT est un correcteur PD (DOC §9.7.6) qui sait le faire.
  - On l'enregistre -> dataset (état, commande).
  - On entraîne un MLP à l'imiter (les 4 briques PyTorch du DOC §7.6).
  - On teste le MLP EN BOUCLE FERMÉE et on compare à l'expert.

C'est exactement le pipeline de Tudor (drone) et le tien (747), en miniature.
Lance :  python toy_cloning.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
np.random.seed(0)

# ----------------------------------------------------------- 1. LA "PHYSIQUE"
# Un chariot : pos' = pos + dt*vel ; vel' = vel + dt*(accel - frottement)
# La commande u est dans [0,1] (comme les moteurs du drone de Tudor) :
# u = 0.5 -> accélération nulle ; u = 1 -> accél max ; u = 0 -> décél max.
DT, A_MAX, FROTTEMENT = 0.05, 2.0, 0.3
N_PAS = 100                                   # 5 s par trajectoire


def dynamique(pos, vel, u):
    accel = (u - 0.5) * 2 * A_MAX - FROTTEMENT * vel
    return pos + DT * vel, vel + DT * accel


# ----------------------------------------------------------- 2. L'EXPERT (PD)
def expert(pos, vel, kp=2.0, kd=1.5):
    accel_voulue = -kp * pos - kd * vel       # rappel + amortissement
    return np.clip(accel_voulue / (2 * A_MAX) + 0.5, 0.0, 1.0)


# ------------------------------------------------- 3. GÉNÉRER LE DATASET
def generer_trajectoires(n_traj):
    """Enregistre l'expert : retourne états (N,2) et commandes (N,1)."""
    etats, commandes = [], []
    for _ in range(n_traj):
        pos, vel = np.random.uniform(-3, 3), np.random.uniform(-1, 1)
        for _ in range(N_PAS):
            u = expert(pos, vel)
            etats.append([pos, vel])
            commandes.append([u])
            pos, vel = dynamique(pos, vel, u)
    return np.array(etats, np.float32), np.array(commandes, np.float32)


print("Génération du dataset (l'expert pilote 2000 fois)...")
X, Y = generer_trajectoires(2000)
print(f"  {len(X)} paires (état, commande)")

# Normalisation min-max vers [0,1] avec des bornes FIXES (DOC §2.3bis !)
# Les MÊMES bornes serviront en boucle fermée -> écrites en dur ici.
X_MIN, X_MAX = np.array([-4.0, -3.0], np.float32), np.array([4.0, 3.0], np.float32)


def normaliser(x):
    return (x - X_MIN) / (X_MAX - X_MIN)


loader = DataLoader(TensorDataset(torch.tensor(normaliser(X)), torch.tensor(Y)),
                    batch_size=64, shuffle=True)

# ------------------------------------------------- 4. LE MODÈLE (Brique 2)
modele = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1), nn.Sigmoid(),           # sortie bornée [0,1] = commande sûre
)

# --------------------------------------- 5. LA BOUCLE D'ENTRAÎNEMENT (Brique 3)
optimiseur = torch.optim.Adam(modele.parameters(), lr=1e-3)
perte = nn.MSELoss()

print("Entraînement...")
for epoque in range(5):
    total = 0.0
    for x, y in loader:
        loss = perte(modele(x), y)
        optimiseur.zero_grad()
        loss.backward()
        optimiseur.step()
        total += loss.item() * len(x)
    print(f"  époque {epoque + 1}/5 : MSE = {total / len(X):.6f}")

# --------------------------------------- 6. LA BOUCLE FERMÉE (Brique 4)
# LE vrai test (DOC §3.3) : le réseau pilote, ses erreurs s'accumulent.
def vol(pilote, pos, vel):
    traj = [(pos, vel)]
    for _ in range(N_PAS * 2):                # on lui laisse 10 s
        u = pilote(pos, vel)
        pos, vel = dynamique(pos, vel, u)
        traj.append((pos, vel))
    succes = abs(pos) < 0.05 and abs(vel) < 0.05
    return succes, np.array(traj)


def pilote_reseau(pos, vel):
    with torch.no_grad():
        x = torch.tensor(normaliser(np.array([pos, vel], np.float32)))
        return float(modele(x))               # MÊME normalisation qu'à l'entraînement


print("Test en boucle fermée (100 départs aléatoires)...")
resultats = {"expert": 0, "réseau": 0}
exemples = []
for i in range(100):
    p0, v0 = np.random.uniform(-3, 3), np.random.uniform(-1, 1)
    ok_e, traj_e = vol(expert, p0, v0)
    ok_r, traj_r = vol(pilote_reseau, p0, v0)
    resultats["expert"] += ok_e
    resultats["réseau"] += ok_r
    if i < 3:
        exemples.append((traj_e, traj_r))

print(f"  expert : {resultats['expert']}/100 réussites")
print(f"  réseau : {resultats['réseau']}/100 réussites   <- ton clone !")

# ------------------------------------------------- 7. TRACÉ (optionnel)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, (traj_e, traj_r) in zip(axes, exemples):
        t = np.arange(len(traj_e)) * DT
        ax.plot(t, traj_e[:, 0], label="expert (PD)")
        ax.plot(t, traj_r[:, 0], "--", label="réseau (clone)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("temps (s)")
    axes[0].set_ylabel("position")
    axes[0].legend()
    fig.suptitle("Behavioral cloning jouet : l'expert et son clone en boucle fermée")
    fig.tight_layout()
    fig.savefig("toy_resultats.png", dpi=120)
    print("Figure sauvée : toy_resultats.png")
except ImportError:
    pass

# --------------------------------------------------------------- Exercices
# 1. Diminue n_traj à 50 : le clone se dégrade -> courbe d'apprentissage (§11.2b).
# 2. Ajoute du bruit sur pos/vel à l'entraînement : le clone devient plus robuste.
# 3. Supprime la Sigmoid : commandes hors [0,1] -> comprends pourquoi on borne.
# 4. Remplace le MLP par un CfC (pip install ncps) : c'est le passage au §7.7.
