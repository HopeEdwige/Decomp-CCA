import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QSpinBox, QLabel)

# --- 1. Votre Algorithme CCA ---
def CCAdecomp(sig, taux):
    # 1. Remove mean over time (axis=1)
    sig = sig - sig.mean(axis=1, keepdims=True)

    # 2. Build delayed versions
    x = sig[:, :-taux]
    y = sig[:, taux:]   # shifted version (correct alignment)

    # 3. QR decompositions (on transposed to get same behavior)
    Q_x, R_x = np.linalg.qr(x.T, mode='reduced')  # Q_x: (n_samples-taux, n_channels)
    Q_y, R_y = np.linalg.qr(y.T, mode='reduced')

    # 4. Singular Value Decomposition
    U, S, Vt = np.linalg.svd(Q_x.T @ Q_y, full_matrices=False)

    # 5. Canonical sources and correlations
    sources = (Q_x @ U).T          # shape (n_components, n_samples - taux)
    autocor = S                    # canonical correlations (singular values)

    # 6. Spatial filters (weights)
    w_x = np.linalg.solve(R_x, U)  # shape (n_channels, n_components)

    return sources, w_x, autocor


# --- 2. L'Interface Graphique ---
class CCAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interface de Décomposition CCA (Style MUedit)")
        self.resize(1000, 800)

        # Génération d'un signal fictif pour l'exemple (4 canaux, 2000 points)
        self.generate_dummy_data()

        self.initUI()
        self.plot_raw_signals()

    def generate_dummy_data(self):
        """Génère des signaux sinusoides mélangés à du bruit (ex: EMG brut)."""
        t = np.linspace(0, 10, 2000)
        s1 = np.sin(2 * np.pi * 1.5 * t)
        s2 = np.sin(2 * np.pi * 3.0 * t)
        
        # Matrice de mélange arbitraire pour simuler des canaux
        mixing_matrix = np.array([[0.8, 0.2], [0.4, 0.5], [0.3, 0.8], [0.9, 0.1]])
        sources_pures = np.vstack([s1, s2])
        
        self.signal = np.dot(mixing_matrix, sources_pures)
        # Ajout de bruit
        self.signal += 0.3 * np.random.randn(*self.signal.shape)

    def initUI(self):
        # Widget principal et Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Panneau de contrôle (En haut) ---
        control_layout = QHBoxLayout()
        
        self.btn_run = QPushButton("Lancer la décomposition CCA")
        self.btn_run.clicked.connect(self.run_cca)
        
        lbl_taux = QLabel("Paramètre 'taux' :")
        self.spin_taux = QSpinBox()
        self.spin_taux.setMinimum(1)
        self.spin_taux.setMaximum(100)
        self.spin_taux.setValue(1)

        control_layout.addWidget(self.btn_run)
        control_layout.addWidget(lbl_taux)
        control_layout.addWidget(self.spin_taux)
        control_layout.addStretch() # Pousse les éléments à gauche

        main_layout.addLayout(control_layout)

        # --- Graphiques (PyQtGraph) ---
        # Configuration globale de pyqtgraph (fond blanc, lignes noires = plus scientifique)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # Graphe pour le signal original
        self.plot_raw = pg.PlotWidget(title="Signal Brut (Multicanal)")
        self.plot_raw.setLabel('left', 'Amplitude')
        self.plot_raw.setLabel('bottom', 'Temps (échantillons)')
        main_layout.addWidget(self.plot_raw)

        # Graphe pour les sources décomposées
        self.plot_sources = pg.PlotWidget(title="Sources CCA (Composantes)")
        self.plot_sources.setLabel('left', 'Amplitude')
        self.plot_sources.setLabel('bottom', 'Temps (échantillons)')
        main_layout.addWidget(self.plot_sources)

    def plot_raw_signals(self):
        """Affiche les canaux originaux avec un décalage vertical pour bien les voir."""
        self.plot_raw.clear()
        n_channels = self.signal.shape[0]
        offset_step = np.max(np.abs(self.signal)) * 1.5 # Espace entre les courbes
        
        for i in range(n_channels):
            # On décale chaque canal pour ne pas qu'ils se superposent
            shifted_signal = self.signal[i, :] + (i * offset_step)
            self.plot_raw.plot(shifted_signal, pen=pg.mkPen(color=(0, 114, 189), width=1.5))

    def run_cca(self):
        """Exécute l'algorithme et met à jour le graphe du bas."""
        taux = self.spin_taux.value()
        
        # Appel de votre fonction
        sources, w_x, autocor = CCAdecomp(self.signal, taux)
        
        # Affichage des sources
        self.plot_sources.clear()
        n_components = sources.shape[0]
        offset_step = np.max(np.abs(sources)) * 1.5
        
        # Couleurs différentes pour différencier les composantes
        colors = [(217, 83, 25), (237, 177, 32), (126, 47, 142), (119, 172, 48)]
        
        for i in range(n_components):
            color = colors[i % len(colors)]
            shifted_source = sources[i, :] + (i * offset_step)
            self.plot_sources.plot(shifted_source, pen=pg.mkPen(color=color, width=1.5))

# --- 3. Lancement de l'Application ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CCAMainWindow()
    window.show()
    sys.exit(app.exec_())