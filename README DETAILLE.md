# Proximity Matrix Finder

Module Python pour la détection d'objets dans des vidéos et le calcul de matrices de proximité entre objets, permettant de trouver les frames où des configurations spécifiques de proximité sont atteintes.

## 🚀 Démarrage rapide

### 1. Vérifier la configuration

```bash
# Exécuter le script de diagnostic
python test_setup.py
```

Ce script vérifie :
- ✓ Tous les modules Python nécessaires sont installés
- ✓ Les chemins vers le modèle SAM2 sont corrects
- ✓ Le device de calcul (CUDA/MPS/CPU) est disponible
- ✓ Le modèle SAM2 se charge correctement

### 2. Configurer les chemins

Éditez `config.py` pour ajuster les chemins selon votre installation :

```python
# Structure automatique (recommandé)
SAM2_CHECKPOINT = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = SAM2_ROOT / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"

# OU utiliser des chemins absolus
SAM2_CHECKPOINT = "C:/SUP/PIE/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "C:/SUP/PIE/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
```

### 3. Lancer un exemple

```bash
python exemple_utilisation.py
```

## 📁 Structure recommandée

```
SAM2/
├── checkpoints/
│   └── sam2.1_hiera_large.pt
├── configs/
│   └── sam2.1/
│       └── sam2.1_hiera_l.yaml
└── projet_final/
    ├── config.py                          # Configuration des chemins
    ├── proximity_matrix_finder.py         # Module principal
    ├── exemple_utilisation.py             # Exemples
    ├── test_setup.py                      # Script de diagnostic
    ├── frames_9A/                         # Vos frames vidéo
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    └── README.md
```

## Fonctionnalités

- **Détection d'objets** : Utilise SAM2 pour détecter des objets à partir de points de référence
- **Propagation de masques** : Propage automatiquement les masques détectés sur toutes les frames
- **Calcul de proximité** : Calcule les distances entre objets pour chaque frame
- **Recherche de critères** : Trouve les frames correspondant à des configurations de proximité spécifiques

## Structure du module

```
proximity_matrix_finder.py       # Module principal
exemple_utilisation.py            # Exemples d'utilisation
README.md                         # Ce fichier
```

## Installation des dépendances

```bash
pip install numpy torch scipy pillow tqdm
# + Installation de SAM2 selon les instructions officielles
```

## Utilisation

### Méthode 1 : Fonction wrapper simple

```python
from proximity_matrix_finder import process_video_with_criteria
import numpy as np

# Définir les points de détection initiale
prompts = {
    1: (np.array([[100, 200], [150, 250]]), np.array([1, 1])),  # Objet 1
    2: (np.array([[300, 400]]), np.array([1])),                  # Objet 2
    3: (np.array([[500, 600]]), np.array([1])),                  # Objet 3
}

# Définir les critères de proximité recherchés
matrices_criteres = {
    9: np.array([[0, 1, 1],   # Tous les objets en contact
                 [1, 0, 1],
                 [1, 1, 0]])
}

# Exécuter le traitement
results = process_video_with_criteria(
    video_dir="./frames",
    sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    prompts=prompts,
    matrices_criteres=matrices_criteres,
    distance_threshold=5.0,  # Distance en pixels
    verbose=True
)

# Résultats : {9: 83} signifie que le critère 9 est atteint à la frame 83
```

### Méthode 2 : Contrôle granulaire avec la classe

```python
from proximity_matrix_finder import ProximityMatrixFinder
import numpy as np

# Initialisation
finder = ProximityMatrixFinder(
    sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    device="cuda",
    verbose=True
)

# Charger la vidéo
inference_state, frame_names = finder.initialize_inference_state("./frames")

# Détecter les objets
prompts = {
    1: (np.array([[100, 200]]), np.array([1])),
    2: (np.array([[300, 400]]), np.array([1])),
}

masks = finder.detect_objects_on_frame(
    inference_state=inference_state,
    frame_idx=0,
    prompts=prompts
)

# Propager les masques
video_segments = finder.propagate_masks(
    inference_state=inference_state,
    frame_names=frame_names
)

# Calculer les matrices de proximité
obj_ids = list(masks.keys())
proximity_matrices = finder.compute_proximity_matrices(
    video_segments=video_segments,
    obj_ids=obj_ids,
    distance_threshold=5.0
)

# Rechercher les critères
matrices_criteres = {
    1: np.array([[0, 1], [1, 0]])  # Les 2 objets en contact
}

results = finder.find_criteria_frames(
    matrices_criteres=matrices_criteres,
    proximity_matrices=proximity_matrices
)
```

## Paramètres

### `process_video_with_criteria`

| Paramètre | Type | Description | Défaut |
|-----------|------|-------------|--------|
| `video_dir` | str/Path | Répertoire contenant les frames | - |
| `sam2_checkpoint` | str | Chemin vers le checkpoint SAM2 | - |
| `model_cfg` | str | Chemin vers la config SAM2 | - |
| `prompts` | dict | Points de détection {obj_id: (coords, labels)} | - |
| `matrices_criteres` | dict | Critères {critere_id: matrice} | - |
| `reference_frame_idx` | int | Frame pour la détection initiale | 0 |
| `distance_threshold` | float | Distance max pour un contact (pixels) | 5.0 |
| `surface_threshold` | int | Surface minimale d'une composante | 100 |
| `device` | str/None | Device ('cuda', 'mps', 'cpu') | None |
| `verbose` | bool | Afficher les informations | True |

### Format des prompts

```python
prompts = {
    obj_id: (coordonnées, labels)
}
```

- **obj_id** : Identifiant unique de l'objet (int)
- **coordonnées** : Array numpy de forme (N, 2) avec les positions [x, y]
- **labels** : Array numpy de forme (N,) avec 1 pour point positif, 0 pour négatif

Exemple :
```python
prompts = {
    1: (np.array([[100, 200], [150, 250]]), np.array([1, 1])),  # 2 points positifs
    2: (np.array([[300, 400]]), np.array([1])),                  # 1 point positif
}
```

### Format des matrices de critères

Les matrices sont des arrays numpy booléens ou binaires (0/1) de dimension (n_objets, n_objets).

- **Symétrique** : matrix[i][j] == matrix[j][i]
- **Diagonale nulle** : matrix[i][i] == 0
- **1 ou True** : Les objets i et j sont en contact
- **0 ou False** : Les objets i et j ne sont pas en contact

Exemple pour 3 objets :
```python
# Tous les objets en contact
matrice_1 = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# Seuls les objets 0 et 1 en contact
matrice_2 = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
])
```

## Exemples de cas d'usage

### 1. Détection de collision en chirurgie

Détecter le moment où des instruments chirurgicaux entrent en contact :

```python
# Définir les instruments
prompts = {
    1: (np.array([[x1, y1]]), np.array([1])),  # Pince
    2: (np.array([[x2, y2]]), np.array([1])),  # Scalpel
    3: (np.array([[x3, y3]]), np.array([1])),  # Aiguille
}

# Critère : tous les instruments en contact
matrices_criteres = {
    "tous_en_contact": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
}

results = process_video_with_criteria(...)
```

### 2. Suivi de l'évolution temporelle

Suivre différentes étapes d'une procédure :

```python
matrices_criteres = {
    "etape_1": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),  # Séparés
    "etape_2": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),  # 1-2 contact
    "etape_3": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),  # Tous en contact
}
```

### 3. Validation de procédure

Vérifier qu'une séquence d'actions a bien été réalisée :

```python
# Rechercher plusieurs critères
results = process_video_with_criteria(...)

# Vérifier la chronologie
if results["etape_1"] < results["etape_2"] < results["etape_3"]:
    print("✓ Procédure correcte")
else:
    print("✗ Procédure incorrecte")
```

## Structure de sortie

La fonction `process_video_with_criteria` retourne un dictionnaire :

```python
{
    critere_id: frame_idx  # ou None si non trouvé
}
```

Exemple :
```python
{
    9: 83,      # Critère 9 trouvé à la frame 83
    10: None,   # Critère 10 non trouvé
    11: 125     # Critère 11 trouvé à la frame 125
}
```

## Optimisation des performances

### Utilisation de GPU

```python
# Force l'utilisation de CUDA
results = process_video_with_criteria(
    ...,
    device="cuda"
)
```

### Réduction de la résolution

Pour des vidéos très longues, réduire la résolution des frames peut accélérer le traitement :

```bash
# Avec ffmpeg
ffmpeg -i input.mp4 -vf scale=640:480 output.mp4
```

### Ajustement des seuils

- **distance_threshold** : Augmenter pour détecter des proximités plus larges
- **surface_threshold** : Augmenter pour ignorer les petits artefacts

## Résolution de problèmes

### Erreur: "No such file or directory: checkpoint"

**Symptôme**: `FileNotFoundError` lors du chargement du modèle

**Solutions**:
1. Exécutez `python test_setup.py` pour diagnostiquer le problème
2. Vérifiez que le checkpoint existe bien dans `SAM2/checkpoints/`
3. Ajustez les chemins dans `config.py`:
   ```python
   # Option 1: Chemin relatif (si vous êtes dans projet_final)
   SAM2_CHECKPOINT = Path("../checkpoints/sam2.1_hiera_large.pt")
   
   # Option 2: Chemin absolu (plus fiable)
   SAM2_CHECKPOINT = "C:/SUP/PIE/sam2/checkpoints/sam2.1_hiera_large.pt"
   ```
4. Ou passez directement le chemin absolu à la fonction:
   ```python
   results = process_video_with_criteria(
       sam2_checkpoint="C:/SUP/PIE/sam2/checkpoints/sam2.1_hiera_large.pt",
       model_cfg="C:/SUP/PIE/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
       ...
   )
   ```

### Objets non détectés

- Vérifier que les coordonnées des prompts sont correctes
- Ajouter plus de points positifs pour l'objet
- Diminuer `surface_threshold`

### Critère non trouvé

- Vérifier que la matrice critère est correcte
- Ajuster `distance_threshold`
- Vérifier que la configuration existe réellement dans la vidéo

### Mémoire insuffisante

- Utiliser un batch de frames plus petit
- Réduire la résolution des frames
- Utiliser CPU au lieu de GPU si nécessaire

## Licence

Ce module est fourni à des fins éducatives et de recherche.

## Auteur

Généré depuis un notebook Jupyter pour faciliter l'utilisation modulaire.
