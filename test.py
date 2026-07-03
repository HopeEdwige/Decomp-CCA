# import scipy.io as sio
# data = sio.loadmat('s1.mat')
# print(data.keys()) # Pour voir comment accéder aux données

#import scipy.io as sio

# Charger le fichier
#data = sio.loadmat('s1.mat') # ou s2.mat

# Accéder au signal EMG (64 canaux ou autre, vérifiez la forme)
#emg_signal = data['dsfilt_emg'] 

# Accéder à la cinématique des doigts
#kinematics = data['finger_kinematics']

#print(f"Forme du signal EMG : {emg_signal.shape}")

import scipy.io as sio
import matplotlib.pyplot as plt
data = sio.loadmat('s1.mat')
emg_data = data['dsfilt_emg']

print(f"Type de l'objet EMG : {type(emg_data)}")
print(f"Forme de l'objet EMG : {emg_data.shape}")


# Si c'est un tableau d'objets (cell array en Matlab), il faut extraire le contenu
# Essayez d'accéder au premier élément si c'est une liste
print(emg_data[0,0].shape)

#import matplotlib.pyplot as plt

# Visualiser les 2000 premiers points du premier canal
#plt.plot(emg_data[:2000, 0])
#plt.title("Extrait du signal EMG (Canal 1)")
#plt.show()