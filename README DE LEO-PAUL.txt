-----------------------------------------------------
		Contenu du projet
-----------------------------------------------------

Voilà le dossier global du projet, dans lequel on peut implémenter tout ce sur quoi on a travaillé en parallèle. Il n'y a besoin de rien d'autre que le contenu du dossier pour exécuter le projet.


Organisation : 

- Les fichiers .py : 
	config.py (qui contient toutes les variables "en dur" du projet)
	evaluation_procedure.py
	frame_extractor.py
	proximity_matrix_finder.py
	test_setup.py
	... Les autres fichiers dont les fichiers de vérification de critère (critere_xx.py)

- Les requirements dans requiremens.txt qui sont automatiquement installés lors de l'utilisation de notre projet.

- Le dossier data qui contient :
	video : le dossier dans lequel déposer la vidéo à analyser
	frames : le dossier dans lequel seront déposées les frames par le script frame_extractor_v3.py

- Les fichiers de fonctionnement de sam (modèles large, small et tiny pour pouvoir modifier le modèle utilisé en cours de route)

- Le README.txt qui détaille l'utilisation précise du projet. 


-----------------------------------------------------
		Comment lancer le projet ?
-----------------------------------------------------

Le projet se lance actuellement depuis le script exemple_utilisation.py. Celui-ci fait deux choses : 
- Il extrait les frames à partir de la video (pas besoin de spécifier le nom de la vidéo, juste de la mettre dans le dossier video)
- Il appelle la fonction process_video_with_criteria() qui va appeler le reste des scripts du projet, dont notamment proximity_matrix_finder.py.

proximity_matrix_finder traite les frames : il récupère les positions initiales de chaque objet donnée en entrée (à modifier pour incorporer la détection automatique par Yolo plus tard) et propage la détection des masques dans l'ensemble des frames. Puis il calcule les matrices de proximité pour chaque frame et identifie pour chaque critère, la frame qui correspond au début du critère et la renvoie.

critere_xx.py devra ensuite prendre en entrée le numéro de la frame à partir duquel il commence à évaluer la procédure, et renverra une évaluation (note, compte rendu pdf, anything else…). Il faudra aussi trouver un moyen de spécifier à critere_xx.py à quelle frame l'action s'arrête. 




















