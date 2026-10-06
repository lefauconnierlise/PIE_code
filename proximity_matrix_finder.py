"""
Module pour la détection d'objets dans des vidéos et le calcul de matrices de proximité.

Ce module permet de :
1. Détecter des objets sur une frame de référence
2. Propager les masques de détection sur toutes les frames
3. Calculer les matrices de proximité entre objets pour chaque frame
4. Trouver les frames correspondant à des critères de proximité donnés

"""

import os
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import label
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import shutil
import tempfile
import matplotlib.pyplot as plt
import time # a supprimer après debug
import h5py
from utils import save_variable_to_file, load_variable_from_file


class ProximityMatrixFinder:
    """
    Classe principale pour gérer la détection d'objets et le calcul des matrices de proximité.
    """
    
    def __init__(self, 
                 sam2_checkpoint: str,
                 model_cfg: str,
                 device: Optional[str] = None,
                 verbose: bool = True, 
                 chunk_size: int = 500):
        """
        Initialise le détecteur avec le modèle SAM2.
        
        Args:
            sam2_checkpoint: Chemin vers le checkpoint du modèle SAM2
            model_cfg: Chemin vers le fichier de configuration du modèle
            device: Device à utiliser ('cuda', 'mps', 'cpu' ou None pour auto-détection)
            verbose: Afficher les informations de progression
        """
        self.verbose = verbose
        self.chunk_size = chunk_size
        self._setup_environment()
        self.device = self._setup_device(device)
        self.predictor = self._load_sam2_model(sam2_checkpoint, model_cfg)
        
    def _setup_environment(self):
        """Configure l'environnement système."""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
    def _setup_device(self, device: Optional[str] = None) -> torch.device:
        """
        Configure le device pour les calculs.
        
        Args:
            device: Device spécifique ou None pour auto-détection
            
        Returns:
            torch.device configuré
        """
        if device is not None:
            selected_device = torch.device(device)
        elif torch.cuda.is_available():
            selected_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            selected_device = torch.device("mps")
        else:
            selected_device = torch.device("cpu")
        
        if self.verbose:
            print(f"Using device: {selected_device}")
        
        # Configuration spécifique CUDA
        if selected_device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif selected_device.type == "mps" and self.verbose:
            print("\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA.")
            
        return selected_device
    
    def _load_sam2_model(self, checkpoint: str, model_cfg: str):
        """
        Charge le modèle SAM2.
        
        Args:
            checkpoint: Chemin vers le checkpoint
            model_cfg: Chemin vers la configuration
            
        Returns:
            Predictor SAM2 chargé
        """

        from sam.configs.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            model_cfg,
            checkpoint,
            device=self.device,
        )
        
        if self.verbose:
            print("SAM2 model loaded successfully")
            
        return predictor
    
    @staticmethod
    def keep_components_above_threshold(mask: np.ndarray, 
                                       threshold: int = 100) -> np.ndarray:
        """
        Garde uniquement les composantes connexes d'un masque dont la surface 
        est supérieure au seuil.
        
        Args:
            mask: Masque binaire numpy array
            threshold: Seuil minimal de surface en pixels
            
        Returns:
            Masque filtré
        """
        if not mask.any():
            return mask
            
        labeled_array, num_features = label(mask)
        bincount = np.bincount(labeled_array.ravel())
        valid_indices = np.where(bincount[1:] >= threshold)[0] + 1
        
        if len(valid_indices) == 0:
            return np.zeros_like(mask, dtype=bool)
            
        return np.isin(labeled_array, valid_indices)
    
    @staticmethod
    def compute_min_distance_kdtree(mask1: np.ndarray, 
                                   mask2: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Calcule la distance minimale entre deux masques en utilisant KDTree.
        
        Args:
            mask1: Premier masque binaire
            mask2: Deuxième masque binaire
            
        Returns:
            Tuple (distance_minimale, coords_mask1_plus_proche, coords_mask2_plus_proche)
            Retourne (np.inf, None, None) si un masque est vide
        """
        coords1 = np.argwhere(mask1)
        coords2 = np.argwhere(mask2)
        
        if len(coords1) == 0 or len(coords2) == 0:
            return np.inf, None, None
        
        # Construction du KDTree pour optimiser la recherche
        tree = cKDTree(coords2)
        distances, indices = tree.query(coords1)
        
        # Trouve le point le plus proche
        min_idx_in_coords1 = np.argmin(distances)
        min_distance = distances[min_idx_in_coords1]
        
        coords1_closest = coords1[min_idx_in_coords1]
        coords2_closest = coords2[indices[min_idx_in_coords1]]
        
        return min_distance, coords1_closest, coords2_closest
    
    def load_video_frames(self, frames_dir: Union[str, Path]) -> Tuple[List[str], List[str]]:
        """
        Charge les noms des frames depuis un répertoire.
        
        Args:
            frames_dir: Chemin vers le répertoire contenant les frames
            
        Returns:
            Tuple (liste des noms de fichiers, liste des chemins complets)
        """
        frames_dir = Path(frames_dir)
        
        # Supporte différents formats d'images
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        frame_paths = []
        
        for ext in extensions:
            frame_paths.extend(sorted(frames_dir.glob(f'*{ext}')))
        
        frame_names = [fp.name for fp in frame_paths]
        frame_paths_str = [str(fp) for fp in frame_paths]
        
        if self.verbose:
            print(f"Loaded {len(frame_names)} frames from {frames_dir}")
            
        return frame_names, frame_paths_str
    
    def initialize_inference_state(self, 
                                   video_dir: Union[str, Path]) -> Tuple[object, List[str]]:
        """
        Initialise l'état d'inférence pour une vidéo.
        
        Args:
            video_dir: Répertoire contenant les frames de la vidéo
            
        Returns:
            Tuple (inference_state, liste des noms de frames)
        """
        video_dir = str(video_dir)
        frame_names, _ = self.load_video_frames(video_dir)
        
        inference_state = self.predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=True,    # Charge les images sur la RAM CPU
            offload_state_to_cpu=True     # Décharge les états d'inférence si besoin
            )
        self.predictor.reset_state(inference_state)
  
        return inference_state, frame_names
    
    def detect_objects_on_frame(self,
                               inference_state: object,
                               frame_idx: int,
                               prompts: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> Dict[int, np.ndarray]:
        """
        Détecte des objets sur une frame spécifique à partir de prompts (points).
        
        Args:
            inference_state: État d'inférence SAM2
            frame_idx: Index de la frame de référence
            prompts: Dictionnaire {obj_id: (coords, labels)} où:
                     - coords: array de forme (N, 2) avec les coordonnées [x, y]
                     - labels: array de forme (N,) avec les labels (1=positif, 0=négatif)
        
        Returns:
            Dictionnaire {obj_id: masque_binaire}
        """
        collected_masks = {}
        
        for obj_id, (points, labels) in prompts.items():
            # Ajoute les points pour l'objet
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
            
            # --- FIX : TROUVER LE BON INDEX ---
            # On cherche à quel index se trouve notre 'obj_id' dans la liste renvoyée
            obj_idx = out_obj_ids.index(obj_id)
            
            # On récupère le masque correspondant à cet index
            mask = (out_mask_logits[obj_idx] > 0.0).cpu().numpy()
            mask = self.keep_components_above_threshold(mask, threshold=100)
            collected_masks[obj_id] = mask
        
        if self.verbose:
            print(f"Detected {len(collected_masks)} objects on frame {frame_idx}")
            
        print("Masks collecté dans la fonction detect_objects_on_frame:")
        for obj_id, mask in collected_masks.items():
            print(f"  - Obj {obj_id}: shape {mask.shape}")
        
        return collected_masks
    
    def propagate_masks_sequentially(self,
                                    video_dir: str,
                                    frame_names: List[str],
                                    obj_ids: List[int],
                                    initial_masks: Dict[int, np.ndarray],
                                    chunk_size: int,
                                    surface_threshold: int = 100) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Version séquentielle de la propagation pour économiser la RAM.
        L'utilisateur ne voit qu'une seule barre de progression continue.
        """
        all_video_segments = {}
        last_masks = initial_masks
        num_frames = len(frame_names)
        
        # 1. Initialisation des deux barres fixes
        # position=0 est la barre du haut, position=1 celle du dessous
        pbar_total = tqdm(total=num_frames, desc="📊 Total Vidéo", unit="fr", position=0, leave=True)
        pbar_chunk = tqdm(total=chunk_size, desc="📦 Batch Actuel", unit="fr", position=1, leave=True)

        for start_idx in range(0, num_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, num_frames)
            current_chunk_len = end_idx - start_idx
            chunk_frames = frame_names[start_idx:end_idx]
            
            # Mise à jour de la configuration de la barre de chunk pour ce batch
            pbar_chunk.reset(total=current_chunk_len)
            pbar_total.set_description(f"📊 Global (Frames {start_idx}-{end_idx}/{num_frames})")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Création des liens/copies (logique identique)
                for f_name in chunk_frames:
                    src = os.path.join(video_dir, f_name)
                    dst = os.path.join(temp_dir, f_name)
                    if os.name == 'nt': shutil.copy2(src, dst)
                    else: os.symlink(os.path.abspath(src), dst)

                inference_state = self.predictor.init_state(
                    video_path=temp_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True
                )
                
                # Injection des masques
                for obj_id, mask in last_masks.items():
                    mask_input = torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask
                    mask_input = mask_input.squeeze()
                    if mask_input.dim() > 2: mask_input = mask_input[0]

                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        mask=mask_input
                    )

                # 2. Propagation : on itère sur le générateur
                # Note: On s'assure que le predictor ne print rien ici
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    masks = (out_mask_logits > 0.0).float()
                    actual_frame_idx = start_idx + out_frame_idx
                    
                    all_video_segments[actual_frame_idx] = {
                        out_obj_id: self.keep_components_above_threshold(
                            masks[i].squeeze().cpu().numpy(), 
                            threshold=surface_threshold
                        )
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                    
                    # Mise à jour synchronisée des deux barres
                    pbar_total.update(1)
                    pbar_chunk.update(1)
                    pbar_chunk.set_postfix({"id_reel": actual_frame_idx})

                # Passage de témoin
                last_frame_in_chunk = max(all_video_segments.keys())
                last_masks = all_video_segments[last_frame_in_chunk]

                self.predictor.reset_state(inference_state)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        # Nettoyage final
        pbar_chunk.close()
        pbar_total.set_description("✅ Propagation terminée")
        pbar_total.close()
        
        # Un petit print pour remettre le curseur après les barres fixes
        print("\n" * 2) 
        return all_video_segments
    
    def compute_proximity_matrices(self,
                                  video_segments: Dict[int, Dict[int, np.ndarray]],
                                  obj_ids: List[int],
                                  distance_threshold: float = 5.0) -> Dict[int, np.ndarray]:
        """
        Calcule les matrices de proximité entre objets pour chaque frame.
        
        Args:
            video_segments: Dictionnaire {frame_idx: {obj_id: masque}}
            obj_ids: Liste des IDs d'objets détectés
            distance_threshold: Distance maximale (en pixels) pour considérer un contact
            
        Returns:
            Dictionnaire {frame_idx: matrice_proximité_booléenne}
        """
        n_objects = len(obj_ids)
        proximity_matrices = {}
        
        if self.verbose:
            print(f"\nComputing proximity matrices...")
            print(f"Distance threshold: {distance_threshold} pixels, {n_objects} objects")
        
        iterator = video_segments.items()
        if self.verbose:
            iterator = tqdm(iterator, 
                          desc="Processing frames",
                          unit="frame")
        
        for frame_idx, masks in iterator:
            # Matrice symétrique de proximité (booléenne)
            proximity_matrix = np.zeros((n_objects, n_objects), dtype=bool)
            
            # Calcule les distances pour toutes les paires d'objets
            for i, obj_i in enumerate(obj_ids):
                for j in range(i + 1, n_objects):
                    obj_j = obj_ids[j]
                    
                    if obj_i in masks and obj_j in masks:
                        min_dist, _, _ = self.compute_min_distance_kdtree(
                            masks[obj_i], 
                            masks[obj_j]
                        )
                        is_contact = min_dist <= distance_threshold
                        
                        # Remplit la matrice symétriquement
                        proximity_matrix[i, j] = is_contact
                        proximity_matrix[j, i] = is_contact
            
            proximity_matrices[frame_idx] = proximity_matrix
        
        if self.verbose:
            print(f"✓ Proximity matrices computed for {len(proximity_matrices)} frames")
            
        return proximity_matrices
    
    @staticmethod
    def find_matching_frame(target_matrix: np.ndarray,
                           proximity_matrices: Dict[int, np.ndarray]) -> Optional[int]:
        """
        Trouve la frame dont la matrice de proximité correspond exactement à la matrice cible.
        
        Args:
            target_matrix: Matrice de proximité cible (critère)
            proximity_matrices: Dictionnaire {frame_idx: matrice_proximité}
            
        Returns:
            Index de la frame correspondante ou None si aucune correspondance
        """
        for frame_idx, matrix in proximity_matrices.items():
            if np.array_equal(target_matrix, matrix):
                return frame_idx
        return None
    

    def find_criteria_frames_v2(self,
                             matrices_criteres: Dict[int, List[np.ndarray]],
                             proximity_matrices: Dict[int, np.ndarray]):
        """
        Trouve les frames de début et de fin pour chaque critère.
        Suppose que matrices_criteres contient toujours des listes [matrice_debut, matrice_fin].
        
        Args:
            matrices_criteres: Dict {critere_id: [matrice_debut, matrice_fin]}
            proximity_matrices: Dict {frame_idx: matrice_proximité}
            
        Returns:
            Dictionnaire {critere_id: [frame_debut, frame_fin]}
        """
        results = {}
        
        # Tri des frames pour garantir l'ordre chronologique
        sorted_frames = sorted(proximity_matrices.keys())
        
        for critere_id, targets in matrices_criteres.items():
            
            # On récupère directement les deux matrices
            target_start = targets[0]
            target_end = targets[1]
            
            start_frame = None
            end_frame = None
            
            # 1. Recherche de la frame de DÉBUT
            for frame_idx in sorted_frames:
                if np.array_equal(target_start, proximity_matrices[frame_idx]):
                    start_frame = frame_idx
                    break # On prend la toute première occurrence
            
            # 2. Recherche de la frame de FIN (Uniquement si le début a été trouvé)
            if start_frame is not None:
                # On ne cherche que dans les frames APRES le début
                start_index_in_list = sorted_frames.index(start_frame)
                remaining_frames = sorted_frames[start_index_in_list + 1:]
                
                for frame_idx in remaining_frames:
                    if np.array_equal(target_end, proximity_matrices[frame_idx]):
                        end_frame = frame_idx
                        break # On prend la première occurrence de fin
                
                results[critere_id] = [start_frame, end_frame]
                
                if self.verbose:
                    print(f"✓ Critère {critere_id}: Début={start_frame}, Fin={end_frame}")
                    if end_frame is None:
                        print(f"  ⚠️ Attention: Fin du critère {critere_id} non trouvée !")
            else:
                results[critere_id] = [None, None]
                if self.verbose:
                    print(f"✗ Critère {critere_id}: Début non trouvé")
        
        return results


def process_video_with_criteria(
    video_dir: Union[str, Path],
    sam2_checkpoint: str,
    model_cfg: str,
    prompts: Dict[int, Tuple[np.ndarray, np.ndarray]],
    matrices_criteres: Dict[int, List[np.ndarray]],
    reference_frame_idx: int = 0,
    distance_threshold: float = 5.0,
    surface_threshold: int = 100,
    device: Optional[str] = None,
    verbose: bool = True
):# -> Dict[int, List[Optional[int]]]:
    """
    Fonction principale pour traiter une vidéo et trouver les frames correspondant aux critères.
    
    Args:
        video_dir: Répertoire contenant les frames de la vidéo
        sam2_checkpoint: Chemin vers le checkpoint SAM2
        model_cfg: Chemin vers la configuration SAM2
        prompts: Dictionnaire {obj_id: (coords, labels)} pour la détection initiale
        matrices_criteres: Dictionnaire {critere_id: [matrice_proximité_cible_debut, matrice_proximité_cible_fin]}
        reference_frame_idx: Index de la frame de référence pour la détection
        distance_threshold: Distance max (pixels) pour considérer un contact
        surface_threshold: Surface minimale pour garder une composante
        device: Device à utiliser ('cuda', 'mps', 'cpu' ou None)
        verbose: Afficher les informations de progression
        
    Returns:
        Dictionnaire {critere_id: frame_idx} avec les frames correspondant aux critères
        
    Example:
        >>> prompts = {
        ...     1: (np.array([[100, 200], [150, 250]]), np.array([1, 1])),
        ...     2: (np.array([[300, 400]]), np.array([1])),
        ... }
        >>> matrices_criteres = {
        ...     9: np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        ... }
        >>> results = process_video_with_criteria(
        ...     video_dir="./frames",
        ...     sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        ...     model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        ...     prompts=prompts,
        ...     matrices_criteres=matrices_criteres
        ... )
    """
    # Initialisation du finder
    finder = ProximityMatrixFinder(sam2_checkpoint, model_cfg, device, verbose)
    
    # 1. Charger la liste des noms de fichiers sans initialiser SAM2 sur tout le dossier
    frame_names, _ = finder.load_video_frames(video_dir)
    
    # 2. Initialiser un état temporaire juste pour la détection initiale (ex: frame 0)
    # On crée un mini-dossier avec juste la frame de référence pour ne pas saturer la RAM
    with tempfile.TemporaryDirectory() as temp_dir:
        ref_frame_name = frame_names[reference_frame_idx]
        shutil.copy2(os.path.join(video_dir, ref_frame_name), os.path.join(temp_dir, ref_frame_name))
        
        # Initialisation légère uniquement pour la détection
        tmp_state = finder.predictor.init_state(
            video_path=temp_dir, 
            offload_video_to_cpu=True,
            offload_state_to_cpu=True)
        detected_masks = finder.detect_objects_on_frame(
            inference_state=tmp_state,
            frame_idx=0, # C'est la frame 0 du dossier temporaire
            prompts=prompts
        )
        finder.predictor.reset_state(tmp_state)
        del tmp_state

    ####### DEBUGAGE A SUPPRIMER ########
    # Après la boucle : affiche UNE SEULE figure avec tous les masques superposés
    # fig, ax = plt.subplots(figsize=(9, 6))
    # ax.imshow(Image.open(os.path.join(video_dir, frame_names[0])))
    # for obj_id, mask in detected_masks.items():
    #     show_mask(mask, ax, obj_id=obj_id)
    #     if obj_id in prompts:
    #         show_points(*prompts[obj_id], ax)

    # ax.axis('off')
    # plt.title(f"frame {frame_names[0]} - tous les masques détectés")
    # plt.show()
    ####### DEBUGAGE A SUPPRIMER ########
    
    

    t0 = time.time()
    # 3. Lancer la propagation séquentielle (qui gère ses propres init_state par blocs)
    obj_ids = list(detected_masks.keys())
    video_segments = finder.propagate_masks_sequentially(
        video_dir=video_dir,
        frame_names=frame_names,
        obj_ids=obj_ids,
        initial_masks=detected_masks,
        chunk_size=finder.chunk_size # A diminuer selon les capacités de RAM, ou augmenter pour accélérer si RAM suffisante
    )
    print(f"\n⏱️  Propagation séquentielle terminée en {time.time() - t0:.2f} secondes")

    # Sauvegarde des video_segments
    save_variable_to_file(video_segments, 'video_segments')

    # Calcul des matrices de proximité
    proximity_matrices = finder.compute_proximity_matrices(
        video_segments=video_segments,
        obj_ids=obj_ids,
        distance_threshold=distance_threshold
    )
    
    # Sauvegarde des proximity_matrices
    save_variable_to_file(proximity_matrices, 'proximity_matrices')
    
    # Recherche des frames correspondant aux critères
    results = finder.find_criteria_frames_v2(
        matrices_criteres=matrices_criteres,
        proximity_matrices=proximity_matrices
    )
    
    # Sauvegarde des results
    save_variable_to_file(results, 'results')
    
    # Affichage d'une image RGB brute avec les masks coloriés
    # ID 13 -> Rouge, ID 7 -> Vert, ID 9 -> Bleu
    critere_id, frame_idx = next(iter(results.items()))
    if frame_idx[0] is not None:
        masks = video_segments.get(frame_idx[0], {})
    else:
        masks = {}    # Récupère les dimensions de l'image depuis un mask
    if masks:
        first_mask = next(iter(masks.values()))
        h, w = first_mask.shape
        
        # Crée une image RGB
        rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Assigne les couleurs selon l'ID
        color_map = {
            13: [255, 0, 0],    # Rouge
            7: [0, 255, 0],     # Vert
            9: [0, 0, 255]      # Bleu
        }
        
        for obj_id, mask in masks.items():
            if obj_id in color_map:
                #print(f"Coloring object {obj_id} with color {color_map[obj_id]}")
                color = color_map[obj_id]
                # Applique la couleur où le mask est True
                for c in range(3):
                    rgb_image[mask > 0, c] = color[c]
        
        # Affiche l'image
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(rgb_image)
        ax.axis('off')
        plt.title(f"Frame {frame_idx} - Masques en couleurs RGB\n(13=Rouge, 7=Vert, 9=Bleu)")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Pas de masques trouvés pour le frame {frame_idx}")

    return results, proximity_matrices, video_segments


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Module proximity_matrix_finder chargé avec succès!")
    print("\nUtilisation:")
    print("from proximity_matrix_finder import process_video_with_criteria")
    print("\nOu utilisez la classe ProximityMatrixFinder pour plus de contrôle.")












########## à Supprimerrrrrrr ##########
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Couleurs très distinctes : Bleu, Orange, Vert, Rouge, Violet, Marron, Rose, Gris, Jaune, Cyan
        # On utilise 'tab10' qui est beaucoup plus saturée que 'tab20'
        cmap = plt.get_cmap("tab10")
        
        # Si tu as plus de 10 objets, on peut boucler ou utiliser une liste manuelle encore plus large
        # Ici on utilise un modulo 10 pour rester sur les couleurs flashies
        cmap_idx = 0 if obj_id is None else int(obj_id) % 10
        color = np.array([*cmap(cmap_idx)[:3], 0.6])

    # Gestion de la forme du masque (pour accepter [H, W] ou [1, H, W])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
########## à Supprimerrrrrrr ##########