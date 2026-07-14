# Notes — Fausses détections YOLO (« pistes téléportées ») : détection et question ouverte

*Notes du 02/07/2026, à ressortir au moment de l'entraînement du CfC et de son
évaluation. Statut : **outil en réserve** — la décision de filtrer ou non les
entrées du LNN est une QUESTION EXPÉRIMENTALE (voir §3), pas un acquis.*

## 1. Le phénomène, mesuré (étude domain gap, 648 img/modèle, ZBTJ)

Déplacement des coins entre détections consécutives espacées d'1 s :

| Modèle | médiane | p95 | sauts > 100 px (« téléportations ») | chutes d'aire > 30 % |
|---|---|---|---|---|
| pose_v8 (retenu) | 5.3 px | 31.1 | **12 (≈ 3.2 % des paires)** | 8 |
| seg_loo_flsim (doublure) | 7.6 px | 21.7 | 1 | 27 |
| seg_full / seg_flsim | 7.5 px | 20-28 | 1-3 | 28-31 |

Lecture : le mouvement « normal » est très stable (5-8 px/s) ; la pose
téléporte ~3 % de ses détections (le prix de sa sensibilité), les seg ont
plutôt des instabilités d'aire (l'approximation polygonale du masque qui
« respire »). Les erreurs des deux familles sont **décorrélées**.
Réf. EUCASS Table 2 : 2-5 px/frame à 25 Hz — cohérent avec nos 5-8 px/s.

## 2. L'arsenal de détection (3 couches, du moins cher au plus robuste)

**Couche 1 — géométrie intra-image (une image suffit, aucune base de données) :**
- (a) forme : quadrilatère convexe, non croisé, plus large en bas qu'en haut,
  étiquettes cohérentes (HG au-dessus de BG…) ;
- (b) **test de l'horizon** (le plus puissant) : avec l'attitude IRS, l'horizon
  est connu dans l'image ; une vraie piste est SOUS l'horizon et le **point de
  fuite de ses bords tombe à ± 2-3° de l'horizon**. Piste dans le ciel ou point
  de fuite aberrant = rejet garanti par la géométrie. (C'est le « moniteur de
  sécurité » du DOC §9.6.6 — les angles servent à surveiller, pas à piloter.)

**Couche 2 — cohérence temporelle (l'atout 25 Hz) :**
- (c) gating de saut : rejet si déplacement des coins > seuil depuis la
  dernière détection valide. Seuils CALIBRÉS sur nos mesures : **60-100 px/s**
  (≈ 3-5 px/frame à 25 Hz) ne coupe que les téléportations ;
- (d) monotonie de l'aire : chute > 30 %/s en approche = suspect ;
- (e) mini-filtre α-β par coin en espace image (la version poids plume du
  VAEKF d'EUCASS, sans base de données) : prédiction ; innovation trop grande
  → mesure rejetée, validité dégradée.

**Couche 3 — double-avis :**
- (f) vote pose_v8 × seg_loo_flsim : coins d'accord à < 30 px = solide ;
  désaccord = drapeau de doute. Erreurs décorrélées (cf. §1) → vote efficace.

**Où l'appliquer, le jour venu :** hors ligne (filtrer detections.csv avant la
jointure/l'entraînement) et/ou en vol (transformer le drapeau de validité en
**score de cohérence** 0-1 que le CfC apprend à pondérer). Implémentation
prévue : `runway_vision/filter_detections.py` (couches 1+2, vote en option).

## 3. LA question ouverte : faut-il filtrer du tout ?

Hypothèse (Maxime) : **envoyer la donnée brute NON filtrée dans le LNN** et le
laisser apprendre à gérer les aberrations lui-même. Arguments POUR :
- cohérent avec le parti pris « coins bruts, zéro ingénierie manuelle »
  (DOC §9.8) — un filtre est un prior fait main, avec ses propres bugs ;
- la robustesse au bruit d'entrée est un argument central des LNN (Hasani) et
  le banc CODIT26 **injecte volontairement du bruit à l'entraînement** pour
  durcir le réseau — les téléportations du train pourraient jouer ce rôle ;
- le CfC reçoit la confiance + la validité : les téléportations sont peut-être
  déjà « signées » (confiance plus basse ?) et la mémoire peut les lisser.

Arguments CONTRE (la nuance à ne pas perdre) :
- il y a bruit et bruit : la littérature « noise robustness » couvre les
  petites perturbations ; une téléportation est une **erreur grossière** (3 %
  des frames), plus proche du label corrompu que du bruit gaussien ;
- en boucle fermée, UNE commande aberrante au mauvais moment (arrondi) peut
  suffire — le réseau n'offre aucune garantie, un gate géométrique si.

**Décision proposée : en faire une ABLATION** (l'expérience tranche, pas
l'opinion) :
1. entraîner le CfC sur données brutes vs données filtrées (couches 1+2),
   toutes choses égales ;
2. comparer : MSE val, et surtout boucle fermée (taux de réussite, écarts) ;
3. dans les DEUX cas, garder la couche 1b (horizon) comme **moniteur de
   sécurité en vol** — surveiller n'est pas filtrer : le réseau décide, la
   géométrie borde.

Vérification préalable rapide (avant même l'ablation) : la confiance des
téléportations est-elle plus basse que celle des vraies détections ? (10 lignes
de pandas sur domain_gap_detections.csv — si oui, le réseau a déjà l'info.)
