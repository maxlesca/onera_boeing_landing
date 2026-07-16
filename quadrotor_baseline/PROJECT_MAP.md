# PROJECT MAP — qui fait quoi dans ce repo

*Repo = fork de [TudorAvarvarei/LNN_behavioural_cloning_quadrotor](https://github.com/TudorAvarvarei/LNN_behavioural_cloning_quadrotor)
(remote `upstream`, rebase possible) + le travail du stage par-dessus.
Convention : un dossier par méthode, le harnais partagé à la racine.*

## Vue d'ensemble

```
onera_boeing_landing/
├── train.py, test.py, Simulator_*.py, utils/   ← HARNAIS (Tudor) : drone + commun
├── train_landing_cfc.py (+ .yaml)              ← entraînement ATTERRISSAGE (stage)
├── runway_vision/                               ← méthode YOLO→CfC : perception (stage)
├── dataset_preparation/                         ← outillage données, objectifs 3-4 (stage)
├── examples/                                    ← exercices pédagogiques (stage)
├── Yolo_models_LARD_V2/                         ← submodule : poids YOLO (deel-ai, intouché)
├── checkpoints/, configs/                       ← modèles entraînés + leurs configs figées
└── (futur) end_to_end/                          ← méthode CNN bout-en-bout (approche B)
```

Hors repo (volontairement — trop lourd ou privé) : `../datasets/` (images, npz),
`../resultats/` (rapports, figures, sorties YOLO), `../DOC_*.md` et
`../GUIDE_*.md` (les documents du stage, gitignorés).

## La racine — le harnais d'entraînement (Tudor) et ses points d'entrée

| Fichier | Origine | Rôle |
|---|---|---|
| `train.py` | Tudor | Entraîne un contrôleur **drone** depuis `train_config.yaml` → checkpoint + config figée. Contient aussi des helpers réutilisés partout (`_set_seed`, nommage des checkpoints). |
| `train_config.yaml` | Tudor | Config du problème **drone** (hover) : labels d'entrée/sortie, architecture (`model.type`), dataloader. |
| `test.py` + `test_config.yaml` | Tudor | Évaluation **boucle ouverte** d'un checkpoint (MSE, ablations). |
| `Simulator_random_start.py` | Tudor | Test **boucle fermée** drone : départs aléatoires (le test le plus dur). |
| `Simulator_start_dataset.py` | Tudor | Boucle fermée, départs du dataset (+ warm-up). |
| `Simulator_race_drone.py` | Tudor | Boucle fermée, passage de portes. |
| `simulator_config.yaml` | Tudor | Config des simulateurs (checkpoint à charger, intégration, seuils de réussite). |
| **`train_landing_cfc.py`** | stage | Entraîne le CfC **atterrissage** : (coins YOLO + capteurs avion) → gouvernes. Variante minimale de `train.py` : seule la préparation des données change, tout le reste est réutilisé tel quel. |
| **`train_landing_config.yaml`** | stage | Config du problème **atterrissage** (même schéma que celui de Tudor) : chemins detections.csv/npz, fenêtres (`seq_len`/`stride`), architecture CfC. |
| `README.md` | Tudor | Le README d'origine (non modifié — conflits de rebase). |
| `PROJECT_MAP.md` | stage | Ce fichier. |
| `.gitmodules`, `.gitignore` | — | Submodule LARD ; exclusions (datasets, npz, zip, docs privés `DOC_*`/`GUIDE_*`). |

## `utils/` — les briques partagées (le cœur réutilisable)

| Fichier | Origine | Rôle |
|---|---|---|
| `model_builder.py` | Tudor | **L'usine à modèles** : `build_controller_network(config, in, out)` construit cfc/ltc/ncp/ctrnn/gru/lstm/mlp + blocs conv/mlp optionnels. Utilisée par TOUS les entraînements. |
| `lightning.py` | Tudor | `Lightning_Model` : la boucle train/val/test (MSE boucle ouverte, gestion du canal temps, rollout pas-à-pas au test). Agnostique au véhicule. |
| `data.py` | Tudor | Données **drone** : `get_data` (npz hover → tableaux normalisés), `DatasetController` (wrapper PyTorch, réutilisé par l'atterrissage), `transform_to_sequence`. |
| **`landing_data.py`** | stage | Données **atterrissage** : `get_landing_data` = jointure detections.csv × landing npz, canaux vision (8 coins + confiance + validité, politique hold-last), fenêtres par run. Équivalent de `get_data`. |
| `normalization_limits.py` | Tudor | Bornes min/max globales du **drone** (l'atterrissage porte les siennes DANS ses npz). |
| `liquid_networks.py` | Tudor | Implémentations CFC/LTC/ConvCfC/MLPCfC (sur la lib `ncps`). |
| `standard_networks.py` | Tudor | RNN/CT-RNN/GRU/LSTM. |
| `feedforward.py` | Tudor | Le MLP baseline. |
| `networks.py`, `networks_LNN.py` | Tudor | Briques communes des réseaux. |
| `ablation.py` | Tudor | Retrait de groupes de features (études d'observation partielle). |
| `config.py` | Tudor | Lecture/écriture YAML, création de dossiers. |
| `graphics.py`, `animation.py` | Tudor | Tracés et animations des trajectoires drone. |

## `runway_vision/` — méthode YOLO→CfC : la perception (stage)

| Fichier | Rôle |
|---|---|
| `yolo_runway_corners.py` | Image(s) → **4 coins de piste étiquetés** (HG/HD/BD/BG) + confiances → `detections.csv` + images annotées. Deux modes : 1 étage (pose ou seg) / 2 étages (detect → crop → coins, architecture LARD 2.0). |
| `analyze_domain_gap.py` | Compare plusieurs modèles YOLO sur les mêmes images : taux de détection/confiance vs temps de vol par run (courbes + CSV). A servi à retenir `pose_v8`. |
| `false_detection_filtering.md` | Notes en réserve : les « pistes téléportées » mesurées (~3 % pose), l'arsenal de filtrage en 3 couches, et la question ouverte « filtrer ou laisser le LNN gérer le brut » (→ ablation à faire). |

## `dataset_preparation/` — outillage données, objectifs 3-4 (stage)

| Fichier | Rôle |
|---|---|
| `check_dataset.py` | **Bilan de santé** d'un dataset d'images (dossier OU .zip, sans extraction) : cadence, index, images corrompues/gelées, détection automatique des overlays → rapport. |
| `crop_images.py` | Rogne les bordures Windows (barres MSFS/tâches), détection automatique des bandes, jamais en place → a produit `dataset_sans_barres` (1920×991, c_y −47). |
| `csv_to_npz.py` | CSV d'états → npz **prêt-à-entraîner** : split PAR RUN, lignes NaN écartées, `--sans-ils` (exclut les colonnes ILS), bornes de normalisation calculées sur le train seul et embarquées. |
| `extract_rosbag.py` | Squelette rosbag → images + CSV + contrôle de synchro (à adapter aux topics réels — datasets futurs). |
| `CONVENTIONS_TEMPLATE.md` | À remplir avec Tudor : horloges, unités, repères, caméra, versions (chaque case ☐ = une question). |
| `run_metadata_TEMPLATE.yaml` | Fiche de métadonnées par run (aéroport, météo, heure, résultat…) — la base de la mesure de couverture (objectif 4). |

## `examples/` — pédagogie (stage)

| Fichier | Rôle |
|---|---|
| `toy_cloning.py` | Le behavioral cloning complet sur un chariot 1D (~1 min CPU) : expert PD → dataset → MLP → boucle fermée. Exécuté : clone 100/100. |
| `toy_results.png` | Sa figure (expert vs clone). |

## `Yolo_models_LARD_V2/` — submodule (deel-ai, AGPL, **jamais modifié/renommé**)

Poids YOLO pré-entraînés sur LARD_V2 (`git lfs pull` pour les télécharger).
Vérifiés : `yolo_v8_models/yolov8pose_best.pt` = **pose, kpt_shape [4,2] = les
4 coins — le modèle RETENU** ; `yolov8detect_*` = boîtes (étage 1) ;
`yolo_v11_models/*segN*` = segmentation (FULL/par-source/leave-one-out, entrée
1024) — `LOO_FLSim` = la doublure ; `piano_calibration_model/` = détection du
« piano » (calibration, objectif 3).

## `checkpoints/` et `configs/` — les modèles entraînés

Chaque entraînement dépose `nom_epoch=…_val_loss=….ckpt` dans `checkpoints/` et
sa config exacte figée dans `configs/` (même nom) : un modèle se recharge
toujours avec SA config. Les checkpoints présents = ceux de Tudor (drone).

## Les deux chaînes d'usage, en une ligne chacune

**Drone (Tudor)** : `train_config.yaml` → `train.py` → checkpoint → `test.py`
(boucle ouverte) → `Simulator_*.py` (boucle fermée).

**Atterrissage (stage)** : images → [`crop_images.py`] → [`yolo_runway_corners.py`
--n 1, pose_v8] → detections.csv ; CSV d'états → [`csv_to_npz.py --sans-ils`] →
npz ; puis `train_landing_config.yaml` → `train_landing_cfc.py` (jointure via
`utils/landing_data.py`) → checkpoint. Boucle fermée : à venir (SCHEMIN).
