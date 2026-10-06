from typing import Union, Dict
import numpy as np
from pathlib import Path
import h5py

import h5py
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

def save_variable_to_file(variable: Union[Dict[Any, Any], np.ndarray], name: str) -> None:
    """
    Sauvegarde une variable dans un fichier HDF5.
    Gère les dicts imbriqués (video_segments), les listes d'entiers (results) 
    et les matrices (proximity_matrices).
    """
    temp_dir = Path("data/temporary")
    temp_dir.mkdir(parents=True, exist_ok=True)
    filename = temp_dir / f"{name}.h5"
    
    def create_safe_dataset(parent, key, data):
        """Aide à créer un dataset avec compression uniquement si c'est une matrice."""
        # On remplace None par -1 pour la compatibilité HDF5
        val = data if data is not None else -1
        
        # On compresse seulement si c'est un array avec plus d'un élément
        if isinstance(val, np.ndarray) and val.size > 1:
            parent.create_dataset(str(key), data=val, compression="gzip")
        else:
            parent.create_dataset(str(key), data=val)

    with h5py.File(filename, 'w') as f:
        if isinstance(variable, dict):
            for key, value in variable.items():
                if isinstance(value, dict):
                    # Cas video_segments : dict[int, dict[int, np.ndarray]]
                    grp = f.create_group(str(key))
                    for subkey, subvalue in value.items():
                        create_safe_dataset(grp, subkey, subvalue)
                
                elif isinstance(value, list):
                    # Cas results : dict[int, list[int]]
                    grp = f.create_group(str(key))
                    for i, item in enumerate(value):
                        create_safe_dataset(grp, i, item)
                
                else:
                    # Cas proximity_matrices : dict[int, np.ndarray]
                    create_safe_dataset(f, key, value)
        else:
            # Si c'est directement un ndarray global
            create_safe_dataset(f, name, variable)
    
    print(f"Variable '{name}' sauvegardée avec succès dans {filename}")


def load_variable_from_file(name: str) -> Union[Dict, np.ndarray]:
    """
    Charge une variable depuis un fichier HDF5 et reconstruit les structures
    (listes avec None, dicts imbriqués, matrices).
    """
    temp_dir = Path("data/temporary")
    filename = temp_dir / f"{name}.h5"
    
    if not filename.exists():
        raise FileNotFoundError(f"Le fichier {filename} n'existe pas.")
    
    with h5py.File(filename, 'r') as f:
        # Cas 1 : Dataset unique (ndarray global)
        if len(f.keys()) == 1 and name in f:
            return f[name][()]
        
        # Cas 2 : Structure de dictionnaire
        data = {}
        for key in f.keys():
            item = f[key]
            # Les clés HDF5 sont toujours des strings, on les repasse en int
            dict_key = int(key) if key.isdigit() else key
            
            if isinstance(item, h5py.Group):
                # On regarde les sous-clés pour savoir si c'est une liste (0, 1, 2...)
                # ou un dictionnaire de segments
                sub_keys = list(item.keys())
                
                if all(k.isdigit() for k in sub_keys):
                    # C'est soit une liste (results), soit un dict de segments (video_segments)
                    # Si les clés sont "0", "1"... et consécutives, c'est probablement une liste
                    sorted_subkeys = sorted(sub_keys, key=int)
                    
                    if name == 'results':
                        # Reconstruction de la liste [start, end] avec gestion du None
                        sublist = []
                        for skey in sorted_subkeys:
                            val = item[skey][()]
                            sublist.append(val if val != -1 else None)
                        data[dict_key] = sublist
                    else:
                        # Reconstruction du sous-dict (ex: video_segments)
                        subdict = {}
                        for skey in sub_keys:
                            subdict[int(skey)] = item[skey][()]
                        data[dict_key] = subdict
                else:
                    # Cas par défaut pour un groupe
                    data[dict_key] = {skey: item[skey][()] for skey in sub_keys}
            else:
                # C'est un dataset direct (ex: proximity_matrices)
                data[dict_key] = item[()]
                
        return data