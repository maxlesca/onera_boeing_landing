# CONVENTIONS.md — la source de vérité du projet (À REMPLIR AVEC TUDOR)

> Étape 0 de la démarche (DOC §10.7). Chaque case vide est une question à
> poser. Tant qu'une case est vide, toute comparaison images ↔ états est
> suspecte (DOC §10.6, problèmes A5/B2/B3/D3).

## 1. Horloges et temps
| Question | Réponse |
|---|---|
| Horloge de référence du dataset (temps sim MSFS ? temps ROS ? temps mur Speedgoat ?) | ☐ |
| Les machines sont-elles synchronisées (NTP/PTP) ? Dérive constatée ? | ☐ |
| `header.stamp` des images = instant d'acquisition ou d'enregistrement ? | ☐ |
| Cadence caméra / cadence états / cadence contrôle | ☐ / ☐ / ☐ |
| Latence de rendu+capture mesurée (étape 3) | ☐ ms |

## 2. Unités et repères
| Grandeur | Unité | Repère / convention |
|---|---|---|
| Positions (lat, lon) | degrés ? radians ? | WGS84 ? |
| Altitude | m ? ft ? | ellipsoïde WGS84 ? MSL/géoïde ? ☐ |
| Vitesses | m/s ? kt ? | NED ? ENU ? corps ? ☐ |
| Angles d'attitude | deg ? rad ? | ordre d'Euler (ψθφ ZYX ?) ☐ |
| Position "avion" | — | centre de masse ? point de référence MSFS ? ☐ |

## 3. Caméra
| Question | Réponse |
|---|---|
| Position de la caméra dans le repère avion (bras de levier x,y,z) | ☐ m |
| Orientation de montage (angles caméra/avion) | ☐ |
| FOV réglé dans MSFS (horizontal ? vertical ?) | ☐ ° |
| Résolution native de capture | 1920×1080 ? ☐ |
| Crop appliqué (haut/bas/gauche/droite) et (c_x, c_y) APRÈS crop | ☐ |
| Intrinsèques calibrées (f_x, f_y, c_x, c_y) + date de calibration | ☐ |

## 4. Environnement
| Question | Réponse |
|---|---|
| Version exacte de MSFS (le décor change avec les mises à jour ! DOC §10.6-D2) | ☐ |
| Source des coordonnées des coins de piste (base officielle ? mesurées dans MSFS ?) | ☐ |
| Écarts base officielle ↔ rendu MSFS déjà constatés (papier EUCASS §9.7.8) | ☐ |
| Version du modèle Simulink / SCHEMIN | ☐ |

## 5. Génération des runs
| Question | Réponse |
|---|---|
| Qui pilote pendant la génération (autopilot ILS ? humain ? replay ?) | ☐ |
| Critère de conservation/rejet d'un run (runs 1,3,5,6 absents : pourquoi ?) | ☐ |
| Fiche de métadonnées par run remplie ? (voir fiche_run_TEMPLATE.yaml) | ☐ |
