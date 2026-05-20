import sys
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
from scipy.signal import find_peaks
import pyqtgraph as pg
import pyqtgraph.exporters  # REQUIS pour l'exportation d'images
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QSpinBox, QLabel, QFileDialog, QMessageBox, QDoubleSpinBox)

# --- 1. Algorithmes Mathématiques ---

def CCAdecomp(sig, taux):
    """Algorithme de décomposition CCA classique."""
    sig = sig - sig.mean(axis=1, keepdims=True)
    x = sig[:, :-taux]
    y = sig[:, taux:]
    
    Q_x, R_x = np.linalg.qr(x.T, mode='reduced')
    Q_y, R_y = np.linalg.qr(y.T, mode='reduced')
    
    U, S, Vt = np.linalg.svd(Q_x.T @ Q_y, full_matrices=False)
    sources = (Q_x @ U).T
    w_x = np.linalg.solve(R_x, U)
    return sources, w_x, S

def detect_spikes(source, std_multiplier=4.0, min_distance=20):
    """Détecte les pics (spikes) basés sur l'écart-type."""
    signal_abs = np.abs(source)
    threshold = np.mean(signal_abs) + (std_multiplier * np.std(signal_abs))
    peaks, _ = find_peaks(signal_abs, height=threshold, distance=min_distance)
    return peaks, threshold


# --- 2. Interface Graphique Principale ---

class CCAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCA Explorer - Décomposition et Détection de Spikes")
        self.resize(1200, 950)
        
        self.signal = None
        self.sources = None
        self.source_offset = 0
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Panneau de contrôle Haut ---
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("📂 Charger un fichier (.csv, .mat)")
        self.btn_load.clicked.connect(self.open_file)
        
        self.btn_run_cca = QPushButton("⚡ Décomposition CCA")
        self.btn_run_cca.clicked.connect(self.run_cca)
        self.btn_run_cca.setEnabled(False)
        
        lbl_taux = QLabel("Taux CCA :")
        self.spin_taux = QSpinBox()
        self.spin_taux.setRange(1, 500)
        self.spin_taux.setValue(1)

        # --- Contrôles pour les Spikes ---
        self.btn_spikes = QPushButton("🎯 Détecter les Spikes")
        self.btn_spikes.clicked.connect(self.run_spike_detection)
        self.btn_spikes.setEnabled(False)
        self.btn_spikes.setStyleSheet("background-color: #d1e7dd; font-weight: bold;")

        lbl_std = QLabel("Seuil (x StdDev) :")
        self.spin_std = QDoubleSpinBox()
        self.spin_std.setRange(1.0, 10.0)
        self.spin_std.setValue(4.0)
        self.spin_std.setSingleStep(0.5)

        # --- Nouveau Bouton d'exportation d'image ---
        self.btn_export = QPushButton("📸 Exporter Image")
        self.btn_export.clicked.connect(self.export_as_image)
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet("background-color: #cff4fc;")
        
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_run_cca)
        control_layout.addWidget(lbl_taux)
        control_layout.addWidget(self.spin_taux)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.btn_spikes)
        control_layout.addWidget(lbl_std)
        control_layout.addWidget(self.spin_std)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.btn_export)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)

        # --- Zone d'affichage des statistiques des Unités Motrices (MUs) ---
        self.stats_layout = QHBoxLayout()
        self.lbl_mu_count = QLabel("Nombre d'Unités Motrices (MUs) identifiées : <b>-</b>")
        self.lbl_mu_count.setStyleSheet("font-size: 13pt; color: #0f5132; padding: 5px; background-color: #e2f0d9; border-radius: 4px;")
        self.stats_layout.addWidget(self.lbl_mu_count)
        self.stats_layout.addStretch()
        main_layout.addLayout(self.stats_layout)

        # --- Graphiques ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.plot_raw = pg.PlotWidget(title="Signaux Importés (Brut - 16 premiers canaux)")
        main_layout.addWidget(self.plot_raw)

        self.plot_sources = pg.PlotWidget(title="Sources Extraites et Spikes (CCA)")
        main_layout.addWidget(self.plot_sources)

    def _find_largest_numeric_2d(self, mat_data):
        """Fouille récursivement les structures MATLAB pour extraire la matrice de signal."""
        largest_size = 0
        best_data = None
        
        def extract_recursive(item):
            nonlocal largest_size, best_data
            if isinstance(item, dict):
                for k, v in item.items():
                    if not str(k).startswith('__'): extract_recursive(v)
            elif isinstance(item, (list, tuple)):
                for v in item: extract_recursive(v)
            elif isinstance(item, np.ndarray):
                if item.dtype.names is not None:
                    for name in item.dtype.names: extract_recursive(item[name])
                elif item.dtype == object:
                    for sub_item in item.flat: extract_recursive(sub_item)
                elif np.issubdtype(item.dtype, np.number):
                    val_sq = np.squeeze(item)
                    if val_sq.ndim >= 3:
                        val_sq = val_sq[0]
                        val_sq = np.squeeze(val_sq)
                    if val_sq.ndim == 2 and max(val_sq.shape) > 100:
                        if val_sq.size > largest_size:
                            largest_size = val_sq.size
                            best_data = val_sq
        extract_recursive(mat_data)
        return best_data

    def open_file(self):
        """Ouvre et lit un fichier CSV ou MAT."""
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un signal", "", "Data Files (*.csv *.mat)")
        if not path: return
        
        try:
            best_data = None
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                best_data = df.values
            elif path.endswith('.mat'):
                try:
                    mat = loadmat(path, simplify_cells=True)
                except TypeError:
                    mat = loadmat(path)
                best_data = self._find_largest_numeric_2d(mat)
            
            if best_data is not None:
                self.signal = best_data if best_data.shape[1] > best_data.shape[0] else best_data.T
                self.btn_run_cca.setEnabled(True)
                self.btn_spikes.setEnabled(False)
                self.btn_export.setEnabled(False)
                self.sources = None
                self.lbl_mu_count.setText("Nombre d'Unités Motrices (MUs) identifiées : <b>-</b>")
                self.plot_raw_signals()
                self.plot_sources.clear()
                QMessageBox.information(self, "Succès", f"Fichier chargé avec succès !\n{self.signal.shape[0]} canaux, {self.signal.shape[1]} échantillons trouvés.")
            else:
                QMessageBox.warning(self, "Avertissement", "Aucune matrice de signal EMG valide trouvée.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier :\n{str(e)}")

    def plot_raw_signals(self):
        self.plot_raw.clear()
        if self.signal is None: return
        offset = np.max(np.abs(self.signal)) * 1.5
        for i in range(min(self.signal.shape[0], 16)): 
            self.plot_raw.plot(self.signal[i] + (i * offset), pen=pg.mkPen('b', width=1))

    def run_cca(self):
        """Exécute la décomposition CCA et affiche les sources."""
        if self.signal is None: return
        self.sources, _, _ = CCAdecomp(self.signal, self.spin_taux.value())
        self.plot_sources.clear()
        self.source_offset = np.max(np.abs(self.sources)) * 1.5
        colors = ['#D95319', '#EDB120', '#7E2F8E', '#77AC30']
        
        for i in range(min(self.sources.shape[0], 8)):
            color = colors[i % len(colors)]
            shifted_source = self.sources[i] + (i * self.source_offset)
            self.plot_sources.plot(shifted_source, pen=pg.mkPen(color=color, width=1.2))
            
        self.btn_spikes.setEnabled(True)
        self.btn_export.setEnabled(True)

    def run_spike_detection(self):
        """Détecte les spikes et calcule mathématiquement le nombre d'Unités Motrices (MUs)."""
        if self.sources is None: return
        self.plot_sources.clear()
        colors = ['#D95319', '#EDB120', '#7E2F8E', '#77AC30']
        std_multiplier = self.spin_std.value()
        
        mu_detected_count = 0  # Compteur d'unités motrices valides
        
        for i in range(min(self.sources.shape[0], 8)):
            color = colors[i % len(colors)]
            source = self.sources[i]
            shifted_source = source + (i * self.source_offset)
            
            self.plot_sources.plot(shifted_source, pen=pg.mkPen(color=color, width=1.2))
            peaks, threshold = detect_spikes(source, std_multiplier=std_multiplier)
            
            # Critère scientifique : Si la source contient des décharges (> 3 spikes pour exclure les faux positifs)
            # alors elle isole une Unité Motrice active distincte.
            if len(peaks) >= 3:
                mu_detected_count += 1
            
            if len(peaks) > 0:
                scatter = pg.ScatterPlotItem(
                    x=peaks, 
                    y=shifted_source[peaks], 
                    size=8, 
                    pen=pg.mkPen(None), 
                    brush=pg.mkBrush(255, 0, 0, 200)
                )
                self.plot_sources.addItem(scatter)
        
        # Mise à jour de l'affichage pour vos tuteurs
        self.lbl_mu_count.setText(f"Nombre d'Unités Motrices (MUs) identifiées après décomposition : <b>{mu_detected_count}</b>")

    def export_as_image(self):
        """Exporte le graphique des sources décomposées en fichier image PNG."""
        if self.sources is None: return
        
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le graphique des sources", "", "Images PNG (*.png)", options=options)
        
        if file_path:
            if not file_path.endswith('.png'):
                file_path += '.png'
            try:
                # Utilisation de l'ImageExporter de pyqtgraph sur le PlotWidget des sources
                exporter = pg.exporters.ImageExporter(self.plot_sources.plotItem)
                # Optionnel : Forcer une bonne résolution (largeur de 1600 pixels pour les rapports)
                exporter.parameters()['width'] = 1600
                exporter.export(file_path)
                QMessageBox.information(self, "Succès", f"Graphique exporté avec succès :\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec de l'exportation de l'image :\n{str(e)}")


# --- 3. Lancement sécurisé ---
if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    window = CCAMainWindow()
    window.show()
    sys.exit(app.exec_())