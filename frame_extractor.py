import cv2
import os
import sys
import shutil

def frame_extractor(video_name=None, frame_start=0, frame_stop=-1, frame_step=1):
    # Détermination des chemins par rapport au script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, 'data', 'video')
    output_folder = os.path.join(script_dir, 'data', 'frames')

    # Création du dossier de sortie
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supprime le contenu du dossier frames s'il existe
    if len(os.listdir(output_folder)) > 0:
        print(f"Suppression de : {len(os.listdir(output_folder))} frames dans {output_folder}")

    for filename in os.listdir(output_folder):
        file_path = os.path.join(output_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Erreur lors de la suppression de {file_path} : {e}')

    # Recherche de la vidéo
    if video_name is None:
        fichiers = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not fichiers:
            print("Erreur : Aucun fichier vidéo trouvé dans data/video")
            return
        video_name = fichiers[0]

    video_path = os.path.join(input_folder, video_name)
    base_name = ""

    if not os.path.exists(video_path):
        print(f"Erreur : Le fichier {video_path} est introuvable.")
        return

    # Lecture de la vidéo
    cap = cv2.VideoCapture(video_path)
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Définition de la borne de fin réelle
    if frame_stop == -1 or frame_stop > total_frames_video:
        actual_stop = total_frames_video
    else:
        actual_stop = frame_stop

    # Positionnement au départ
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    count = frame_start
    saved_count = 0

    print(f"Vidéo : {video_name} (Total: {total_frames_video} frames)")
    print(f"Intervalle d'extraction : [{frame_start} à {actual_stop}] avec un pas de {frame_step}")

    while cap.isOpened():
        # Arrêt si on dépasse la borne de fin
        if count >= actual_stop:
            break

        ret, frame = cap.read()
        if not ret:
            break
        
        # On enregistre si la frame actuelle correspond au pas défini depuis le début
        if (count - frame_start) % frame_step == 0:
            file_name = f"{base_name}{saved_count:06d}.jpg"
            save_path = os.path.join(output_folder, file_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1
        
        count += 1
        
        # Progression basée sur l'intervalle [frame_start, actual_stop]
        range_size = actual_stop - frame_start
        if range_size > 0:
            percent = ((count - frame_start) / range_size) * 100
            print(f"\rProgression : {percent:.1f}% | Frame actuelle : {count}/{actual_stop} | Enregistrées : {saved_count}", end='')
            sys.stdout.flush()

    cap.release()
    print(f"\nTerminé ! {saved_count} frames extraites sur l'intervalle demandé.\n")

if __name__ == "__main__":
    # Exemple : extrait une frame sur deux entre la frame 100 et 500
    frame_extractor(frame_start=100, frame_stop=500, frame_step=2)