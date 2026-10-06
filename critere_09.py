import numpy as np
import cv2
from skimage.morphology import skeletonize
from scipy.signal import savgol_filter



def validate_criterion_9(
    video_segments: dict,
    frame_start: int,
    frame_end: int,
    object_id: int,
    log_lines,
    report_file="rapport_critere_9.txt"
):
    """
    Validate Criterion 9: Needle trajectory
    (only insertion count and redressing, no echography criteria)

    needle_masks: list of SAM2 binary masks (H, W)
    """

    masks = []

    for frame_idx in range(frame_start, frame_end + 1):

        if frame_idx not in video_segments:
            continue  # frame absente

        frame_data = video_segments[frame_idx]

        if object_id not in frame_data:
            continue

        masks.append(frame_data[object_id])

    tip_positions = []
    angles = []

    for mask in masks:
        skel = skeleton_from_mask(mask)

        tip = extract_tip_from_skeleton(skel)
        angle = compute_needle_angle(skel)

        if tip is not None:
            tip_positions.append(tip)
        angles.append(angle)

    single_insertion = validate_single_insertion(tip_positions, log_lines)
    no_redressing = validate_no_redressing(angles, log_lines)

    criterion_9_valid = single_insertion and no_redressing

    # Mini rapport du critère 9
    report_lines = []
    report_lines.append("\n" + "=" * 50)
    report_lines.append("RAPPORT DU CRITÈRE 9 : Trajectoire de l'aiguille")
    report_lines.append("=" * 50)
    report_lines.append(f"Nombre de frames analysées : {len(masks)}")
    report_lines.append(f"Insertion unique (pas de retrait/réinsertion) : {'OUI' if single_insertion else 'NON'}")
    report_lines.append(f"Pas de redressement : {'OUI' if no_redressing else 'NON'}")
    report_lines.append(f"Critère 9 validé : {'OUI' if criterion_9_valid else 'NON'}")
    report_lines.append("=" * 50)

    # affichage console
    for line in report_lines:
        print(line)
        log_lines.append(line)
    

    # écriture dans fichier
    with open(report_file, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")

    return {
        "criterion_9_valid": criterion_9_valid,
        "single_insertion": single_insertion,
        "no_redressing": no_redressing,
        "n_frames": len(masks)
    }

def skeleton_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize a binary mask (SAM2 output).
    """
    mask_bool = mask.astype(bool)
    skeleton = skeletonize(mask_bool)
    return skeleton.astype(np.uint8)


def extract_tip_from_skeleton(skel: np.ndarray) -> np.ndarray:
    """
    Extract needle tip as the skeleton endpoint farthest from centroid.
    """
    ys, xs = np.where(skel > 0)
    points = np.stack([xs, ys], axis=1)

    if len(points) < 2:
        return None

    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    tip = points[np.argmax(distances)]
    return tip


def compute_needle_angle(skel: np.ndarray) -> float:
    """
    Compute dominant needle angle (in degrees) using PCA on skeleton.
    """
    ys, xs = np.where(skel > 0)
    if len(xs) < 5:
        return None

    points = np.stack([xs, ys], axis=1)
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]

    angle = np.arctan2(principal_axis[1], principal_axis[0])
    return np.degrees(angle)






def validate_single_insertion(tip_positions, log_lines, tolerance_px=200):
    """
    Valide qu'il n'y a qu'une seule insertion en calculant la distance 
    cumulée des retraits le long de l'axe principal d'insertion.

    tip_positions: list of (x, y)
    tolerance_px: seuil de distance de retrait totale autorisée (pixels)
    """
    if len(tip_positions) < 10:
        return False

    pts = np.array(tip_positions)

    # 1. Lissage des coordonnées X et Y pour filtrer le bruit de détection
    # On ajuste window_length pour qu'il soit impair et inférieur à la taille des données
    win = min(11, len(pts) if len(pts) % 2 != 0 else len(pts) - 1)
    pts_smooth_x = savgol_filter(pts[:, 0], window_length=win, polyorder=2)
    pts_smooth_y = savgol_filter(pts[:, 1], window_length=win, polyorder=2)
    pts_smooth = np.stack([pts_smooth_x, pts_smooth_y], axis=1)

    # 2. Définir l'axe principal d'insertion (du début au point le plus éloigné)
    # Cela permet de s'affranchir de l'angle de la caméra
    start_point = pts_smooth[0]
    end_point = pts_smooth[-1]
    insertion_vector = end_point - start_point
    norm = np.linalg.norm(insertion_vector)
    
    if norm < 1e-5: # Pas de mouvement détecté
        return False
    
    unit_vector = insertion_vector / norm

    # 3. Calculer les vecteurs de déplacement entre chaque frame
    displacements = np.diff(pts_smooth, axis=0)

    # 4. Projeter chaque déplacement sur l'axe d'insertion
    # Une valeur positive = on avance / Une valeur négative = on retire
    projections = np.dot(displacements, unit_vector)

    # 5. Calculer la distance totale parcourue dans le sens du retrait
    # On ne somme que les valeurs négatives
    retraction_distances = projections[projections < 0]
    total_retraction_dist = np.abs(np.sum(retraction_distances))

    if total_retraction_dist > 0:
        line = f"Distance totale de retrait détectée : {total_retraction_dist:.2f} px sur tolérance de {tolerance_px} px"
        print(line)
        log_lines.append(line)
        
    
    # Renvoie True si le retrait total est inférieur au seuil de tolérance
    return total_retraction_dist < tolerance_px


def validate_no_redressing(angles, log_lines, max_angle_change_deg=8):
    """
    Validate absence of needle redressing.

    angles: list of angles (degrees)
    max_angle_change_deg: tolerated angular variation
    """
    angles = np.array([a for a in angles if a is not None])

    if len(angles) < 5:
        return False

    # On convertit la liste en array numpy
    angles_deg = np.array(angles)

    # 1. On définit la référence (la première frame)
    angle_ref = angles_deg[0]

    # 2. On calcule l'écart de chaque angle par rapport à cette référence
    diffs = angles_deg - angle_ref

    # 3. On ramène ces écarts dans l'intervalle [-90°, 90°]
    # C'est ici que la magie du modulo 180 opère pour supprimer les sauts de PCA
    corrected_diffs = (diffs + 90) % 180 - 90

    # 4. On reconstruit les angles corrigés
    angles_fixed = angle_ref + corrected_diffs

    # Calcul de la variation sur les angles corrigés
    angle_variation = np.abs(np.diff(angles_fixed))
    line = f"Variation maximale d'angle détectée (corrigée) : {np.max(angle_variation):.2f}° sur tolérance de {max_angle_change_deg}°"
    print(line)
    log_lines.append(line)

    # Pour tes résultats finaux, utilise 'angles_fixed'
    return np.max(angle_variation) < max_angle_change_deg