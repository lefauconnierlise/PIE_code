"""
Fichier de configuration pour proximity_matrix_finder.

Ajustez ces chemins selon votre structure de projet.
"""

import os
import sys
import subprocess
import importlib
import re
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
# On considère que `projet_final` est la racine du projet partagé.
SAM2_ROOT = CURRENT_DIR  # projet_final devient la racine

# Dossier local contenant les sources SAM (regrouper ici les fichiers SAM nécessaires)
SAM_DIR = SAM2_ROOT / "sam"

# Chemins vers les modèles SAM2 (pointent vers le dossier `sam` local)
SAM2_CHECKPOINT = SAM_DIR / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = SAM_DIR / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"

# Alternative : chemins absolus (décommentez et adaptez si nécessaire)
# SAM2_CHECKPOINT = "C:/chemin/vers/projet_final/sam/checkpoints/sam2.1_hiera_large.pt"
# SAM2_CONFIG = "C:/chemin/vers/projet_final/sam/configs/sam2.1/sam2.1_hiera_l.yaml"

# Répertoire des frames vidéo (dans projet_final/data/frames)
FRAMES_DIR = SAM2_ROOT / "data" / "frames"

# Alternative : chemin absolu
# FRAMES_DIR = "C:/SUP/PIE/sam2/projet_final/frames_9A"

# Paramètres par défaut
DEFAULT_DISTANCE_THRESHOLD = 10.0  # Distance en pixels pour considérer un contact
DEFAULT_SURFACE_THRESHOLD = 100   # Surface minimale d'une composante en pixels
DEFAULT_REFERENCE_FRAME = 0       # Frame de référence pour la détection
DEFAULT_FRAME_STEP = 10  # Pas d'extraction des frames (ex: 10 = une frame sur 10)

# Vérification de l'existence des fichiers
def verify_paths():
    """Vérifie que tous les chemins nécessaires existent."""
    errors = []
    
    if not SAM2_CHECKPOINT.exists():
        errors.append(f"❌ Checkpoint non trouvé: {SAM2_CHECKPOINT}")
    else:
        print(f"✓ Checkpoint trouvé: {SAM2_CHECKPOINT}")
    
    if not SAM2_CONFIG.exists():
        errors.append(f"❌ Config non trouvée: {SAM2_CONFIG}")
    else:
        print(f"✓ Config trouvée: {SAM2_CONFIG}")
    
    if not FRAMES_DIR.exists():
        errors.append(f"❌ Dossier frames non trouvé: {FRAMES_DIR}")
    else:
        print(f"✓ Dossier frames trouvé: {FRAMES_DIR}")
    
    if errors:
        print("\n⚠️  ERREURS DE CONFIGURATION:")
        for error in errors:
            print(f"  {error}")
        print("\n💡 Veuillez ajuster les chemins dans config.py")
        return False
    
    print("\n✓ Tous les chemins sont valides!")

    # Vérifie et installe les packages listés dans requirements.txt (si présent)
    req_file = SAM2_ROOT / "requirements.txt"
    if req_file.exists():
        print(f"\n🔎 Vérification des dépendances depuis: {req_file}")
        req_status = verify_and_install_requirements(req_file, auto_install=True)
        failed = [k for k, v in req_status.items() if v is False]
        if failed:
            print("\n⚠️  Certains modules n'ont pas pu être importés:")
            for name in failed:
                print(f"  - {name}")
            print("\n💡 Installez-les manuellement ou vérifiez votre connexion réseau.")
            return False
        else:
            print("\n✓ Toutes les dépendances listées ont été importées (ou installées) correctement.")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("VÉRIFICATION DE LA CONFIGURATION")
    print("=" * 60)
    print(f"\nRépertoire actuel: {CURRENT_DIR}")
    print(f"Racine SAM2: {SAM2_ROOT}")
    print()
    verify_paths()


def _normalize_pkg_name(line: str) -> str:
    """Récupère le nom du paquet pip depuis une ligne requirements (enlève versions)."""
    line = line.strip()
    # Remove extras like pkg[extra]==1.2 -> pkg
    line = re.split(r"[\[;]", line)[0]
    # split on version separators
    parts = re.split(r"[<>=!~]+", line)
    return parts[0].strip()


def verify_and_install_requirements(requirements_path: Path, auto_install: bool = True) -> dict:
    """Lit un requirements.txt, tente d'importer chaque package, et installe si manquant.

    Retourne un dict {package_spec: True/False} indiquant si l'import a réussi.
    """
    # Mappages courants paquet -> module d'import
    common_map = {
        "pillow": "PIL",
        "opencv-python": "cv2",
        "opencv-python-headless": "cv2",
        "hydra-core": "hydra",
        "omegaconf": "omegaconf",
        "scikit-image": "skimage",
        "pyyaml": "yaml",
    }

    results = {}
    with open(requirements_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Skip editable installs handled by pip directly
        pkg_spec = line
        pkg_name = _normalize_pkg_name(line).lower()

        # candidates for import names
        candidates = [pkg_name]
        if pkg_name in common_map:
            candidates.insert(0, common_map[pkg_name])

        imported = False
        for mod in candidates:
            try:
                importlib.import_module(mod)
                results[pkg_spec] = True
                imported = True
                break
            except Exception:
                continue

        if imported:
            continue

        # Not importable: try to install via pip if allowed
        if auto_install:
            print(f"→ Installation tentative: {pkg_spec}")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", pkg_spec], check=True)
            except subprocess.CalledProcessError:
                print(f"⚠️  Échec de l'installation pip pour: {pkg_spec}")
                results[pkg_spec] = False
                continue

            # Après installation, retenter l'import
            post_ok = False
            for mod in candidates:
                try:
                    importlib.import_module(mod)
                    post_ok = True
                    break
                except Exception:
                    continue

            results[pkg_spec] = post_ok
            if not post_ok:
                print(f"⚠️  Impossible d'importer {pkg_spec} même après installation.")
        else:
            results[pkg_spec] = False

    return results
