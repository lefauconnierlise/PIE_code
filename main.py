import numpy as np
from proximity_matrix_finder import process_video_with_criteria, ProximityMatrixFinder
from frame_extractor import frame_extractor
from critere_09 import *
from utils import load_variable_from_file
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Importer la configuration
try:
    from config import (
        SAM2_CHECKPOINT,
        SAM2_CONFIG,
        FRAMES_DIR,
        DEFAULT_DISTANCE_THRESHOLD,
        DEFAULT_SURFACE_THRESHOLD,
        DEFAULT_REFERENCE_FRAME,
        DEFAULT_FRAME_STEP,
        verify_paths
    )    
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("⚠️  config.py non trouvé, utilisation des chemins par défaut")




if __name__ == "__main__":

    # Choix de charger depuis les fichiers ou calculer
    load_from_file = input("Voulez-vous charger les données depuis les fichiers sauvegardés ? (y/n): ").lower() == 'y'
    
    if load_from_file:
        try:
            results = load_variable_from_file('results')
            proximity_matrices = load_variable_from_file('proximity_matrices')
            video_segments = load_variable_from_file('video_segments')
            print("Données chargées depuis les fichiers.")
        except FileNotFoundError as e:
            print(f"Erreur: {e}. Passage au calcul.")
            load_from_file = False
    
    if not load_from_file:
        #Extraction des frames si pas déjà fait (pas besoin de nom de vidéo tant qu'il n'y en a qu'une)
        frame_extractor(video_name=None, frame_start=720, frame_stop = -1, frame_step=DEFAULT_FRAME_STEP)

        print("=" * 60)
        print("EXEMPLE temporaire : Recherche d'un critère simple avec process_video_with_criteria")
        print("=" * 60)
        
        # Vérification de la configuration
        if USE_CONFIG:
            print("\n🔍 Vérification des chemins...")
            if not verify_paths():
                raise SystemExit("Configuration invalide")
        
        # Définition des chemins
        if USE_CONFIG:
            checkpoint = str(SAM2_CHECKPOINT)
            config = str(SAM2_CONFIG)
            frames = str(FRAMES_DIR)
        else:
            checkpoint = "./sam/checkpoints/sam2.1_hiera_large.pt"
            config = "./sam/configs/sam2.1/sam2.1_hiera_l.yaml"
            frames = "./data/frames"
        
        # Définition des positions de chaque objet pour la détection initiale
        # Format: {obj_id: (coordonnées, labels)}
        # Coordonnées: array [[x1, y1], [x2, y2], ...]
        # Labels: array [1, 1, ...] (1 = point positif, 0 = point négatif)
        prompts = {
            7: (np.array([[1293, 241]]), np.array([1])), # Sonde
            9: (np.array([[1048, 502]]), np.array([1])), # Phantom
            13: (np.array([[839, 433], [799, 506]]), np.array([1, 1])),# Aiguille complète
        }
        
        # Définition des critères (matrices de proximité cibles)
        # Format: {critere_id: matrice_booléenne}
        # La matrice indique quels objets doivent être en contact
        matrices_criteres = {
            9: [np.array([[0, 1, 1], 
                         [1, 0, 1], 
                         [1, 1, 0]]),
                np.array([[0, 0, 0],  # Sonde : Ne touche plus Fantôme (0), Touche Aiguille (1)
                         [0, 0, 1],  # Fantôme : Ne touche plus Sonde (0), Touche Aiguille (1)
                         [0, 1, 0]])]}  # Aiguille : Touche Sonde (1) et Fantôme (1)
        
        # Exécution du traitement
        results, proximity_matrices, video_segments = process_video_with_criteria(
            video_dir=frames,
            sam2_checkpoint=checkpoint,
            model_cfg=config,
            prompts=prompts,
            matrices_criteres=matrices_criteres,
            reference_frame_idx=0,  # Frame de départ pour la détection
            distance_threshold=DEFAULT_DISTANCE_THRESHOLD,  # Distance max pour considérer un contact (pixels)
            surface_threshold=DEFAULT_SURFACE_THRESHOLD,   # Surface minimale pour une composante
            device=None,  # Auto-détection (cuda, mps, ou cpu)
            verbose=True
        )

    # Affichage des résultats
    print("\n" + "=" * 60)
    print("RÉSULTATS:")
    print("=" * 60)
    for critere_id, frames in results.items():
        # frames est maintenant une liste [frame_debut, frame_fin]
        if frames and any(f is not None for f in frames):
            start_frame, end_frame = frames
            
            # Cas où les deux sont trouvés
            if start_frame is not None and end_frame is not None:
                print(f"Critère {critere_id}: trouvé du frame {start_frame} au frame {end_frame}")
            # Cas où seul le début est trouvé
            elif start_frame is not None:
                print(f"Critère {critere_id}: début trouvé à {start_frame}, fin non trouvée")
            # Cas où seule la fin est trouvée (si possible selon ta logique)
            else:
                print(f"Critère {critere_id}: fin trouvée à {end_frame}, début non trouvé")
        else:
            print(f"Critère {critere_id}: non trouvé")

    ##############################################
    """
    SUITE DE L'EVALUATION DE LA PROCEDURE 
    APPEL DES SCRIPTS critere_xx.py POUR CHAQUE CRITERE
    """
    print("Début d'évaluation du critère 9")
    log_lines = []

    line = f"Critère 9: trouvé du frame {results.get(9, [])[0]} au frame {results.get(9, [])[1]}"
    print(line)
    log_lines.append(line)

    line = "Début d'évaluation du critère 9"
    print(line)
    log_lines.append(line)

    result_09 = validate_criterion_9(
        video_segments,
        results.get(9, [])[0],
        results.get(9, [])[1],
        13,
        log_lines
    )
    