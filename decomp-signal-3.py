import sys
import os
import site
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, welch, butter, filtfilt, iirnotch
from sklearn.cluster import KMeans

# ==========================================
# 1. CORRECTIF DES CHEMINS PYQT5 POUR WINDOWS
# ==========================================
if sys.platform.startswith('win'):
    qt_paths = []
    if hasattr(site, 'getsitepackages'):
        qt_paths.extend(site.getsitepackages())
    qt_paths.append(site.getusersitepackages())
    qt_paths.extend(sys.path)
    for base in qt_paths:
        if not base: continue
        qt_bin = os.path.join(base, 'PyQt5', 'Qt5', 'bin')
        qt_plugins = os.path.join(base, 'PyQt5', 'Qt5', 'plugins')
        if os.path.isdir(qt_bin):
            try: os.add_dll_directory(qt_bin)
            except AttributeError: pass
            os.environ['PATH'] = qt_bin + os.pathsep + os.environ.get('PATH', '')
        if os.path.isdir(qt_plugins):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins
        if os.path.isdir(qt_bin) or os.path.isdir(qt_plugins):
            break

import pyqtgraph as pg
import pyqtgraph.exporters 
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QSpinBox, QLabel, QFileDialog, QMessageBox, QDoubleSpinBox)

# ==========================================
# 2. ALGORITHMES MATHÉMATIQUES
# ==========================================

def CCAdecomp(sig, taux):
    """ LA FONCTION EXACTE DU SUPERVISEUR """
    sig = sig - sig.mean(axis=1, keepdims=True)
    x = sig[:, :-taux]
    y = sig[:, taux:]   
    Q_x, R_x = np.linalg.qr(x.T, mode='reduced')  
    Q_y, R_y = np.linalg.qr(y.T, mode='reduced')
    U, S, Vt = np.linalg.svd(Q_x.T @ Q_y, full_matrices=False)
    sources = (Q_x @ U).T          
    autocor = S                    
    w_x = np.linalg.solve(R_x, U)  
    return sources, w_x, autocor

def preprocess_signal(sig, fs=2048.0, f0=50.0):
    """ Filtres Physiologiques (Passe-bande 20-500Hz + Notch 50Hz) """
    sig_mean = sig.mean(axis=1, keepdims=True)
    x_c = sig - sig_mean
    high_freq = min(500.0, (fs / 2) - 1)
    b_band, a_band = butter(4, [20.0 / (fs / 2), high_freq / (fs / 2)], btype='band')
    x_filt = filtfilt(b_band, a_band, x_c, axis=1)
    b_notch, a_notch = iirnotch(f0, 30.0, fs)
    x_filt = filtfilt(b_notch, a_notch, x_filt, axis=1)
    return x_filt

def extend_signal(sig, extension_factor=5):
    """ Extension spatio-temporelle (Le booster de séparation) """
    if extension_factor <= 1:
        return sig
    nb_channels, nb_samples = sig.shape
    extended_sig = np.zeros((nb_channels * extension_factor, nb_samples - extension_factor + 1))
    for i in range(extension_factor):
        extended_sig[i*nb_channels : (i+1)*nb_channels, :] = sig[:, i : nb_samples - extension_factor + 1 + i]
    return extended_sig

def detect_spikes_kmeans(source, fs=2048.0, min_distance_ms=10.0):
    """ Binarisation K-Means avec calcul du PNR (Métrique MUedit) """
    signal_sq = source ** 2
    min_distance = int((min_distance_ms / 1000.0) * fs)
    
    peaks_candidats, _ = find_peaks(signal_sq, distance=min_distance)
    
    if len(peaks_candidats) < 5:
        return np.array([]), 0, 0

    valeurs_pics = signal_sq[peaks_candidats].reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(valeurs_pics)
    classe_spikes = np.argmax(kmeans.cluster_centers_)
    vrais_peaks = peaks_candidats[kmeans.labels_ == classe_spikes]
    
    if len(vrais_peaks) > 0:
        seuil_effectif = np.sqrt(np.min(valeurs_pics[kmeans.labels_ == classe_spikes]))
        
        # --- CALCUL DU PNR (Pulse-to-Noise Ratio) ---
        spikes_power = np.mean(signal_sq[vrais_peaks])
        mask = np.ones(len(signal_sq), dtype=bool)
        mask[vrais_peaks] = False
        noise_power = np.mean(signal_sq[mask])
        pnr = 10 * np.log10(spikes_power / noise_power) if noise_power > 0 else 0
        # ---------------------------------------------
    else:
        seuil_effectif = 0
        pnr = 0
        
    return vrais_peaks, seuil_effectif, pnr

# ==========================================
# 3. INTERFACE GRAPHIQUE PRINCIPALE
# ==========================================

class CCAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCA Explorer - Décodage MUs (Standard MUedit / PNR / Extension)")
        self.resize(1600, 950)
        
        self.signal = None
        self.sources = None
        self.autocor = None
        self.source_offset = 0
        self.mu_stats = []
        self.mu_spikes = {}   # Pour le Train de Dirac
        self.mu_signals = {}  # Pour les sources continues
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- LIGNE 1 : Import & Traitement ---
        layout_l1 = QHBoxLayout()
        self.btn_load = QPushButton("📂 Charger fichier")
        self.btn_load.clicked.connect(self.open_file)

        self.btn_run_cca = QPushButton("⚡ Décomposition CCA")
        self.btn_run_cca.clicked.connect(self.run_cca)
        self.btn_run_cca.setEnabled(False)
        self.btn_run_cca.setStyleSheet("font-weight: bold;")
        
        # NOUVEAU : FACTEUR D'EXTENSION
        lbl_ext = QLabel("Facteur d'Extension :")
        self.spin_ext = QSpinBox()
        self.spin_ext.setRange(1, 20)
        self.spin_ext.setValue(5)

        lbl_taux = QLabel("Taux CCA :")
        self.spin_taux = QSpinBox()
        self.spin_taux.setRange(1, 500)
        self.spin_taux.setValue(1) # Recommandé à 1 pour la stabilité

        lbl_fs = QLabel("Fréq. Éch. (Hz) :")
        self.spin_fs = QSpinBox()
        self.spin_fs.setRange(100, 20000)
        self.spin_fs.setValue(2048) 
        self.spin_fs.setSingleStep(100)

        layout_l1.addWidget(self.btn_load)
        layout_l1.addWidget(self.btn_run_cca)
        layout_l1.addSpacing(15)
        layout_l1.addWidget(lbl_ext)
        layout_l1.addWidget(self.spin_ext)
        layout_l1.addSpacing(15)
        layout_l1.addWidget(lbl_taux)
        layout_l1.addWidget(self.spin_taux)
        layout_l1.addSpacing(15)
        layout_l1.addWidget(lbl_fs)
        layout_l1.addWidget(self.spin_fs)
        layout_l1.addStretch()

        # --- LIGNE 2 : Filtres Biologiques & PNR ---
        layout_l2 = QHBoxLayout()
        self.btn_spikes = QPushButton("🎯 Extraire & Filtrer MUs (K-Means)")
        self.btn_spikes.clicked.connect(self.run_spike_detection)
        self.btn_spikes.setEnabled(False)
        self.btn_spikes.setStyleSheet("background-color: #d1e7dd; font-weight: bold;")

        lbl_pnr = QLabel("PNR Min (dB) :")
        self.spin_pnr = QDoubleSpinBox()
        self.spin_pnr.setRange(0.0, 50.0)
        self.spin_pnr.setValue(12.0) # Ajuste cette valeur si tu as trop/pas assez d'UMs

        lbl_cov = QLabel("CoV Max (%) :")
        self.spin_cov = QDoubleSpinBox()
        self.spin_cov.setRange(5.0, 100.0)
        self.spin_cov.setValue(35.0)

        lbl_spec = QLabel("Énergie Spectrale Min (%) :")
        self.spin_spec = QDoubleSpinBox()
        self.spin_spec.setRange(0.0, 100.0)
        self.spin_spec.setValue(30.0)

        layout_l2.addWidget(self.btn_spikes)
        layout_l2.addSpacing(15)
        layout_l2.addWidget(lbl_pnr)
        layout_l2.addWidget(self.spin_pnr)
        layout_l2.addSpacing(15)
        layout_l2.addWidget(lbl_cov)
        layout_l2.addWidget(self.spin_cov)
        layout_l2.addWidget(lbl_spec)
        layout_l2.addWidget(self.spin_spec)
        layout_l2.addStretch()

        # --- LIGNE 3 : Déduplication et Export ---
        layout_l3 = QHBoxLayout()
        lbl_sync = QLabel("Taux de Coïncidence Max (%) :")
        self.spin_sync = QDoubleSpinBox()
        self.spin_sync.setRange(5.0, 100.0)
        self.spin_sync.setValue(30.0) 

        self.btn_export_csv = QPushButton("📊 Export Stats (CSV)")
        self.btn_export_csv.clicked.connect(self.export_stats_csv)
        self.btn_export_csv.setEnabled(False)

        self.btn_export_sources = QPushButton("📈 Exporter Sources Continues")
        self.btn_export_sources.clicked.connect(self.export_sources_csv)
        self.btn_export_sources.setEnabled(False)
        self.btn_export_sources.setStyleSheet("background-color: #d0ebff; font-weight: bold; color: #00509e;")

        self.btn_export_spikes = QPushButton("📥 Exporter Train de Dirac (0 et 1)")
        self.btn_export_spikes.clicked.connect(self.export_spikes_csv)
        self.btn_export_spikes.setEnabled(False)
        self.btn_export_spikes.setStyleSheet("background-color: #e2e3e5; font-weight: bold; color: #052c65;")

        self.btn_export_img = QPushButton("📸 Export Image")
        self.btn_export_img.clicked.connect(self.export_as_image)
        self.btn_export_img.setEnabled(False)

        layout_l3.addWidget(lbl_sync)
        layout_l3.addWidget(self.spin_sync)
        layout_l3.addStretch()
        layout_l3.addWidget(self.btn_export_csv)
        layout_l3.addWidget(self.btn_export_sources)
        layout_l3.addWidget(self.btn_export_spikes)
        layout_l3.addWidget(self.btn_export_img)

        main_layout.addLayout(layout_l1)
        main_layout.addLayout(layout_l2)
        main_layout.addLayout(layout_l3)

        # --- Bandeau Statistique ---
        self.stats_layout = QHBoxLayout()
        self.lbl_mu_count = QLabel("MUs Uniques Validées : <b>-</b>")
        self.lbl_mu_count.setStyleSheet("font-size: 14px; color: #2c3e50;")
        self.stats_layout.addWidget(self.lbl_mu_count)
        self.stats_layout.addStretch()
        main_layout.addLayout(self.stats_layout)

        # --- Graphiques ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.plot_raw = pg.PlotWidget(title="Signaux Importés")
        main_layout.addWidget(self.plot_raw)
        self.plot_sources = pg.PlotWidget(title="Sources CCA Validées")
        main_layout.addWidget(self.plot_sources)

    def _find_largest_numeric_2d(self, mat_data):
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
                    if val_sq.ndim >= 3: val_sq = np.squeeze(val_sq[0])
                    if val_sq.ndim == 2 and max(val_sq.shape) > 100:
                        if val_sq.size > largest_size:
                            largest_size = val_sq.size
                            best_data = val_sq
        extract_recursive(mat_data)
        return best_data

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir un signal", "", "Data Files (*.mat)")
        if not path: return
        try:
            mat = loadmat(path, simplify_cells=True)
            best_data = self._find_largest_numeric_2d(mat)
            if best_data is not None:
                self.signal = best_data if best_data.shape[1] > best_data.shape[0] else best_data.T
                self.btn_run_cca.setEnabled(True)
                self.plot_raw.clear()
                self.plot_sources.clear()
                offset = np.max(np.abs(self.signal)) * 1.5
                for i in range(self.signal.shape[0]): self.plot_raw.plot(self.signal[i] + (i * offset), pen='b')
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def run_cca(self):
        if self.signal is None: return
        QApplication.setOverrideCursor(pg.QtCore.Qt.WaitCursor)
        try:
            # 1. Filtre Physiologique
            filt_signal = preprocess_signal(self.signal, fs=self.spin_fs.value())
            
            # 2. Extension Spatio-Temporelle (RÉTABLIE !)
            ext_signal = extend_signal(filt_signal, extension_factor=self.spin_ext.value())
            
            # 3. CCA du Superviseur (sur le signal étendu)
            self.sources, w_x, self.autocor = CCAdecomp(ext_signal, self.spin_taux.value())
            
            self.plot_sources.clear()
            self.source_offset = np.max(np.abs(self.sources)) * 1.5
            for i in range(self.sources.shape[0]):
                self.plot_sources.plot(self.sources[i] + (i * self.source_offset), pen=pg.mkPen(color='gray'))
            self.btn_spikes.setEnabled(True)
        finally:
            QApplication.restoreOverrideCursor()

    def run_spike_detection(self):
        if self.sources is None: return
        self.plot_sources.clear()
        self.mu_stats = []
        self.mu_spikes = {} 
        self.mu_signals = {} 
        
        fs = self.spin_fs.value() 
        candidates = []
        
        for i in range(self.sources.shape[0]):
            source = self.sources[i]
            
            peaks, threshold, pnr = detect_spikes_kmeans(source, fs=fs)
            
            is_valid = False
            cov_isi = 1000 
            fr_hz = 0
            
            f, Pxx = welch(source, fs=fs, nperseg=1024)
            mu_spec = (np.sum(Pxx[(f >= 70) & (f <= 400)]) / np.sum(Pxx)) * 100 if np.sum(Pxx)>0 else 0
            
            # LE FILTRE FINAL : 15 pics min, Énergie OK, et PNR OK
            if len(peaks) >= 15 and mu_spec >= self.spin_spec.value() and pnr >= self.spin_pnr.value():
                isi = np.diff(peaks) 
                valid_isi = isi[(isi > 0.5 * np.median(isi)) & (isi < 1.5 * np.median(isi))]
                if len(valid_isi) >= 3:
                    cov_isi = (np.std(valid_isi) / np.mean(valid_isi)) * 100
                    if cov_isi <= self.spin_cov.value():
                        is_valid = True
                        fr_hz = 1.0 / (np.mean(valid_isi) / fs)
            
            candidates.append({"idx": i, "peaks": peaks, "is_valid": is_valid, "cov": cov_isi, "fr": fr_hz, "spec": mu_spec, "pnr": pnr, "dup": False, "threshold": threshold})

        # Déduplication
        tol = int((fs / 1000.0) * 1.5) 
        sync_th = self.spin_sync.value() / 100.0 
        for i in range(len(candidates)):
            if not candidates[i]["is_valid"] or candidates[i]["dup"]: continue
            for j in range(i + 1, len(candidates)):
                if not candidates[j]["is_valid"] or candidates[j]["dup"]: continue
                pi, pj = candidates[i]["peaks"], candidates[j]["peaks"]
                common = sum(1 for p in pi if np.any(np.abs(pj - p) <= tol))
                if (common / min(len(pi), len(pj))) > sync_th:
                    if candidates[i]["pnr"] > candidates[j]["pnr"]: candidates[j]["dup"] = True
                    else: candidates[i]["dup"] = True

        mu_count = 0
        colors = ['#D95319', '#EDB120', '#7E2F8E', '#77AC30']
        for cand in candidates:
            idx, peaks, threshold = cand["idx"], cand["peaks"], cand["threshold"]
            shifted = self.sources[idx] + (idx * self.source_offset)
            
            if cand["is_valid"] and not cand["dup"]:
                mu_count += 1
                color = colors[mu_count % len(colors)]
                self.plot_sources.plot(shifted, pen=pg.mkPen(color=color, width=1.5))
                self.plot_sources.addItem(pg.ScatterPlotItem(x=peaks, y=shifted[peaks], size=8, pen=None, brush=pg.mkBrush('r')))
                
                line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', style=pg.QtCore.Qt.DashLine))
                line.setPos(threshold + (idx * self.source_offset))
                self.plot_sources.addItem(line)
                
                nom_mu = f"MU_{mu_count}"
                
                self.mu_spikes[nom_mu] = peaks                 
                self.mu_signals[nom_mu] = self.sources[idx]    
                
                self.mu_stats.append({
                    "MU": nom_mu, 
                    "Nb_Spikes": len(peaks), 
                    "Fr_Hz": cand["fr"], 
                    "CoV_%": cand["cov"],
                    "PNR_dB": cand["pnr"],
                    "Correlation_CCA": self.autocor[idx]
                })
            else:
                self.plot_sources.plot(shifted, pen=pg.mkPen(color='#e0e0e0'))

        self.lbl_mu_count.setText(f"MUs Uniques Validées : <b>{mu_count}</b>")
        if mu_count > 0: 
            self.btn_export_csv.setEnabled(True)
            self.btn_export_spikes.setEnabled(True) 
            self.btn_export_sources.setEnabled(True) 
            self.btn_export_img.setEnabled(True)

    def export_stats_csv(self):
        if not self.mu_stats: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Stats", "", "CSV (*.csv)")
        if file_path: pd.DataFrame(self.mu_stats).round(4).to_csv(file_path, index=False, sep=';', decimal=',')

    def export_sources_csv(self):
        if not self.mu_signals: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Exporter Sources Continues (Courbes)", "", "Fichier CSV (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'): file_path += '.csv'
            try:
                df_sources = pd.DataFrame(self.mu_signals)
                df_sources.to_csv(file_path, index=False, sep=';')
                QMessageBox.information(self, "Succès", "Sources continues exportées avec succès !\nChaque colonne représente l'onde continue d'une MU.")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec de l'exportation :\n{str(e)}")

    def export_spikes_csv(self):
        if not self.mu_spikes or self.signal is None: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Exporter Train de Dirac", "", "Fichier CSV (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'): file_path += '.csv'
            try:
                signal_length = self.signal.shape[1] 
                export_dict = {}
                for mu_name, spk in self.mu_spikes.items():
                    dirac_train = np.zeros(signal_length, dtype=int)
                    valid_spk = spk[spk < signal_length]
                    dirac_train[valid_spk] = 1
                    export_dict[mu_name] = dirac_train
                
                df_spikes = pd.DataFrame(export_dict)
                df_spikes.to_csv(file_path, index=False, sep=';')
                QMessageBox.information(self, "Succès", "Train de Dirac exporté avec succès !\nIl est composé uniquement de 0 et de 1, sur toute la longueur du signal.")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", f"Échec de l'exportation :\n{str(e)}")

    def export_as_image(self):
        if self.sources is None: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Image", "", "PNG (*.png)")
        if file_path:
            exporter = pg.exporters.ImageExporter(self.plot_sources.plotItem)
            exporter.parameters()['width'] = 2500 
            exporter.export(file_path)

if __name__ == '__main__':
    app = QApplication.instance() or QApplication(sys.argv)
    window = CCAMainWindow()
    window.show()
    sys.exit(app.exec_())