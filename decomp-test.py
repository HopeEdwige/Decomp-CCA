from scipy.io import loadmat
import h5py

fichier = "signaux-simules/Base_20_MUedit/Normal_rep5.mat" # Mets le bon chemin ici

try:
    mat = loadmat(fichier)
    print("Format : Scipy (Ancien Matlab)")
    for key, val in mat.items():
        if not key.startswith('__'):
            print(f"Variable '{key}' : Type = {type(val)}, Shape/Taille = {getattr(val, 'shape', 'Inconnue')}")
except Exception:
    with h5py.File(fichier, 'r') as f:
        print("Format : h5py (Nouveau Matlab v7.3+)")
        for key in f.keys():
            val = f[key]
            print(f"Variable '{key}' : Shape/Taille = {getattr(val, 'shape', 'Inconnue')}")