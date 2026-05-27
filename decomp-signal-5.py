import sys
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch
import pyqtgraph as pg
import pyqtgraph.exporters 
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QFileDialog, QMessageBox, 
                             QLabel, QCheckBox)

# ==============================================================================
# 1. ALGORITHMES MATHÉMATIQUES (BSS & FILTRAGE)
# ==============================================================================

def CCAdecomp(sig, taux):
    """ Algorithme de base CCA """
    sig = sig - sig.mean(axis=1, keepdims=True)
    x = sig[:, :-taux]
    y = sig[:, taux:]
    
    Q_x, R_x = np.linalg.qr(x.T, mode='reduced')
    Q_y, R_y = np.linalg.qr(y.T, mode='reduced')
    
    U, S, Vt = np.linalg.svd(Q_x.T @ Q_y, full_matrices=False)
    sources = (Q_x @ U).T
    w_x = np.linalg.solve(R_x, U)
    return sources, w_x, S

def sCCA_denoise(sig, fs, f0=50.0, taux=1):
    """ Version spectrale de la CCA (ssCCA). """
    sig_mean = sig.mean(axis=1, keepdims=True)
    x_c = sig - sig_mean
    
    # Filtre Notch pour éliminer le 50 Hz (PLI)
    b_notch, a_notch = iirnotch(f0, 30.0, fs)
    x_filt = filtfilt(b_notch, a_notch, x_c, axis=1)
    
    # Filtre Passe-bande physiologique (20 Hz - 500 Hz)
    high_freq = min(500.0, (fs / 2) - 1)
    b_band, a_band = butter(4, [20.0 / (fs / 2), high_freq / (fs / 2)], btype='band')
    x_filt = filtfilt(b_band, a_band, x_filt, axis=1)
    
    # Décomposition
    sources, w_x, S = CCAdecomp(x_filt, taux)
    return sources, w_x, S

def apply_single_differential(data):
    """ Filtre spatial SD pour retirer le mode commun monopolaire. """
    n_channels, n_samples = data.shape
    sd_data = np.zeros((n_channels - 1, n_samples))
    for i in range(n_channels - 1):
        sd_data[i, :] = data[i+1, :] - data[i, :]
    return sd_data

# ==============================================================================
# 2. INTERFACE GRAPHIQUE (PyQt5 + PyQtGraph)
# ==============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Décomposition HD-sEMG par ssCCA (Mode Benchmark)")
        self.setGeometry(100, 100, 1200, 800)

        self.sig_data = None
        self.fs = None
        self.sources = None
        self.axe_temps = None
        self.mu_stats = []

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout_principal = QVBoxLayout(main_widget)

        layout_controles = QHBoxLayout()
        
        self.btn_charger = QPushButton("1. Charger Fichier .mat")
        self.btn_charger.clicked.connect(self.charger_fichier)
        layout_controles.addWidget(self.btn_charger)

        self.lbl_info = QLabel("Aucun fichier chargé.")
        self.lbl_info.setStyleSheet("color: blue; font-weight: bold;")
        layout_controles.addWidget(self.lbl_info)

        self.cb_sd = QCheckBox("Appliquer Filtre SD (Retire le mode commun)")
        self.cb_sd.setChecked(True) 
        layout_controles.addWidget(self.cb_sd)

        self.btn_decomp = QPushButton("2. Exécuter sCCA")
        self.btn_decomp.clicked.connect(self.executer_decomposition)
        self.btn_decomp.setEnabled(False)
        layout_controles.addWidget(self.btn_decomp)

        self.btn_export_csv = QPushButton("Export Stats (CSV)")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setEnabled(False)
        layout_controles.addWidget(self.btn_export_csv)

        self.btn_export_img = QPushButton("Export Graphique (PNG)")
        self.btn_export_img.clicked.connect(self.export_as_image)
        self.btn_export_img.setEnabled(False)
        layout_controles.addWidget(self.btn_export_img)

        layout_principal.addLayout(layout_controles)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_sources = pg.PlotWidget(title="Sources sCCA extraites (Unités Motrices)")
        self.plot_sources.setLabel('bottom', 'Temps', units='s')
        self.plot_sources.setLabel('left', 'Amplitude')
        self.plot_sources.showGrid(x=True, y=False)
        layout_principal.addWidget(self.plot_sources)

    def charger_fichier(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Ouvrir fichier sEMG", "", "Fichiers MATLAB (*.mat)", options=options)
        
        if file_path:
            try:
                mat_data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
                signal_struct = mat_data['signal']
                
                self.fs = float(signal_struct.fsamp) 
                self.sig_data = signal_struct.data
                
                n_channels, n_samples = self.sig_data.shape
                self.axe_temps = np.arange(n_samples) / self.fs
                
                self.lbl_info.setText(f"Chargé : {n_channels} canaux | fs = {int(self.fs)} Hz")
                self.btn_decomp.setEnabled(True)
                
                self.plot_sources.clear()
                self.plot_sources.setTitle("Aperçu du signal brut (Canal 1)")
                self.plot_sources.plot(self.axe_temps, self.sig_data[0, :], pen='b')

            except Exception as e:
                QMessageBox.critical(self, "Erreur de chargement", str(e))

    def executer_decomposition(self):
        if self.sig_data is None: return
            
        try:
            self.lbl_info.setText("Calcul sCCA en cours...")
            QApplication.processEvents() 
            
            data_to_process = self.sig_data
            if self.cb_sd.isChecked():
                data_to_process = apply_single_differential(self.sig_data)
            
            self.sources, self.w_x, S = sCCA_denoise(data_to_process, fs=self.fs, f0=50.0, taux=1)
            
            seuil_correlation = 0.8
            idx_valides = np.where(S > seuil_correlation)[0]
            
            if len(idx_valides) == 0:
                idx_valides = np.arange(min(10, self.sources.shape[0]))
                
            sources_valides = self.sources[idx_valides, :]
            nb_sources = sources_valides.shape[0]
            
            self.plot_sources.clear()
            self.plot_sources.setTitle(f"Décomposition ssCCA terminée : {nb_sources} UM extraites")
            
            decalage_y = 0
            pas_decalage = np.max(np.abs(sources_valides)) * 1.5
            
            # --- CORRECTION DE LA FORME (SHAPE) ICI ---
            longueur_sources = sources_valides.shape[1]
            axe_temps_plot = self.axe_temps[:longueur_sources]
            
            for i in range(nb_sources):
                signal_decale = sources_valides[i, :] - (i * pas_decalage)
                couleur = (255, 127, 14) if i % 2 == 0 else (31, 119, 180) 
                self.plot_sources.plot(axe_temps_plot, signal_decale, pen=pg.mkPen(color=couleur, width=1.5))
            
            self.lbl_info.setText(f"Succès ! {nb_sources} Unités Motrices isolées.")
            self.btn_export_csv.setEnabled(True)
            self.btn_export_img.setEnabled(True)
            
            self.mu_stats = [{'UM_ID': i+1, 'Correlation_CCA': S[idx]} for i, idx in enumerate(idx_valides)]

        except Exception as e:
            QMessageBox.critical(self, "Erreur de décomposition", str(e))
            self.lbl_info.setText("Erreur.")

    def export_csv(self):
        if not self.mu_stats: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer CSV", "", "Fichier CSV (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'): file_path += '.csv'
            pd.DataFrame(self.mu_stats).round(3).to_csv(file_path, index=False, sep=';', decimal=',') 

    def export_as_image(self):
        if self.sources is None: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer PNG", "", "Images PNG (*.png)", options=options)
        if file_path:
            if not file_path.endswith('.png'): file_path += '.png'
            exporter = pg.exporters.ImageExporter(self.plot_sources.plotItem)
            exporter.parameters()['width'] = 2500 
            exporter.export(file_path)

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())