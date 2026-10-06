"""
Script de test pour vérifier que tout est correctement configuré.
Exécutez ce script avant d'utiliser proximity_matrix_finder.
"""

import sys
from pathlib import Path

def test_imports():
    """Teste que tous les modules nécessaires sont installés."""
    print("=" * 60)
    print("TEST DES IMPORTS")
    print("=" * 60)
    
    modules = {
        'numpy': 'numpy',
        'torch': 'PyTorch',
        'PIL': 'Pillow',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
        'sam': 'SAM (local)'
    }
    
    missing = []
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NON INSTALLÉ")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Modules manquants: {', '.join(missing)}")
        print("Installez-les avec: pip install numpy torch pillow scipy tqdm")
        return False
    
    print("\n✓ Tous les modules sont installés!")
    return True


def test_config():
    """Teste la configuration des chemins."""
    print("\n" + "=" * 60)
    print("TEST DE LA CONFIGURATION")
    print("=" * 60)
    
    try:
        from config import verify_paths
        return verify_paths()
    except ImportError:
        print("⚠️  config.py non trouvé")
        print("Créez un fichier config.py ou utilisez des chemins absolus")
        return False


def test_device():
    """Teste la disponibilité des devices de calcul."""
    print("\n" + "=" * 60)
    print("TEST DES DEVICES")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("○ CUDA non disponible")
        
        if torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) disponible")
        else:
            print("○ MPS non disponible")
        
        print(f"\n→ Device recommandé: ", end="")
        if torch.cuda.is_available():
            print("cuda")
        elif torch.backends.mps.is_available():
            print("mps")
        else:
            print("cpu (⚠️  traitement lent)")
        
        return True
    except Exception as e:
        print(f"✗ Erreur lors du test: {e}")
        return False


def test_sam2_model():
    """Teste le chargement du modèle SAM2."""
    print("\n" + "=" * 60)
    print("TEST DU MODÈLE SAM2")
    print("=" * 60)
    
    try:
        from config import SAM2_CHECKPOINT, SAM2_CONFIG
        from proximity_matrix_finder import ProximityMatrixFinder
        
        print("Chargement du modèle SAM2...")
        finder = ProximityMatrixFinder(
            sam2_checkpoint=str(SAM2_CHECKPOINT),
            model_cfg=str(SAM2_CONFIG),
            verbose=False
        )
        print("✓ Modèle SAM2 chargé avec succès!")
        return True
        
    except ImportError as e:
        print(f"✗ Erreur d'import: {e}")
        print("Assurez-vous que config.py et proximity_matrix_finder.py sont présents")
        return False
    except FileNotFoundError as e:
        print(f"✗ Fichier non trouvé: {e}")
        print("Vérifiez les chemins dans config.py")
        return False
    except Exception as e:
        print(f"✗ Erreur inattendue: {e}")
        return False


def main():
    """Exécute tous les tests."""
    print("\n" + "🔍 DIAGNOSTIC DU SYSTÈME")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "Configuration": test_config(),
        "Devices": test_device(),
        "Modèle SAM2": test_sam2_model()
    }
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✓ OK" if result else "✗ ÉCHEC"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ SYSTÈME PRÊT!")
        print("Vous pouvez maintenant utiliser proximity_matrix_finder.py")
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("Veuillez corriger les erreurs avant de continuer")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
