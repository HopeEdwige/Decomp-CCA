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

def detect_spikes_kmeans(source, fs=2048.0, min_distance_ms=10.0):
    """ Binarisation par K-Means (State of the Art - Sans seuil manuel) """
    signal_sq = source ** 2
    min_distance = int((min_distance_ms / 1000.0) * fs)
    
    peaks_candidats, _ = find_peaks(signal_sq, distance=min_distance)
    
    if len(peaks_candidats) < 5:
        return np.array([]), 0

    valeurs_pics = signal_sq[peaks_candidats].reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(valeurs_pics)
    classe_spikes = np.argmax(kmeans.cluster_centers_)
    vrais_peaks = peaks_candidats[kmeans.labels_ == classe_spikes]
    
    if len(vrais_peaks) > 0:
        seuil_effectif = np.sqrt(np.min(valeurs_pics[kmeans.labels_ == classe_spikes]))
    else:
        seuil_effectif = 0
        
    return vrais_peaks, seuil_effectif

# ==========================================
# 3. INTERFACE GRAPHIQUE PRINCIPALE
# ==========================================

class CCAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCA Explorer - Décodage MUs & Déduplication (State-of-the-Art)")
        self.resize(1500, 950)
        
        self.signal = None
        self.sources = None
        self.autocor = None
        self.source_offset = 0
        self.mu_stats = []
        self.mu_spikes = {} 
        
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
        
        lbl_taux = QLabel("Taux CCA :")
        self.spin_taux = QSpinBox()
        self.spin_taux.setRange(1, 500)
        self.spin_taux.setValue(1)

        lbl_fs = QLabel("Fréq. Éch. (Hz) :")
        self.spin_fs = QSpinBox()
        self.spin_fs.setRange(100, 20000)
        self.spin_fs.setValue(2048) 
        self.spin_fs.setSingleStep(100)

        layout_l1.addWidget(self.btn_load)
        layout_l1.addWidget(self.btn_run_cca)
        layout_l1.addWidget(lbl_taux)
        layout_l1.addWidget(self.spin_taux)
        layout_l1.addSpacing(20)
        layout_l1.addWidget(lbl_fs)
        layout_l1.addWidget(self.spin_fs)
        layout_l1.addStretch()

        # --- LIGNE 2 : Filtres Biologiques ---
        layout_l2 = QHBoxLayout()
        self.btn_spikes = QPushButton("🎯 Extraire & Dédupliquer MUs (K-Means)")
        self.btn_spikes.clicked.connect(self.run_spike_detection)
        self.btn_spikes.setEnabled(False)
        self.btn_spikes.setStyleSheet("background-color: #d1e7dd; font-weight: bold;")

        lbl_cov = QLabel("CoV Max (%) :")
        self.spin_cov = QDoubleSpinBox()
        self.spin_cov.setRange(5.0, 100.0)
        self.spin_cov.setValue(30.0)

        lbl_spec = QLabel("Énergie Spectrale Min (%) :")
        self.spin_spec = QDoubleSpinBox()
        self.spin_spec.setRange(0.0, 100.0)
        self.spin_spec.setValue(30.0)

        layout_l2.addWidget(self.btn_spikes)
        layout_l2.addWidget(lbl_cov)
        layout_l2.addWidget(self.spin_cov)
        layout_l2.addWidget(lbl_spec)
        layout_l2.addWidget(self.spin_spec)
        layout_l2.addStretch()

        # --- LIGNE 3 : Déduplication et Export ---
        layout_l3 = QHBoxLayout()
        lbl_sync = QLabel("Taux de Coïncidence Max (Déduplication %) :")
        self.spin_sync = QDoubleSpinBox()
        self.spin_sync.setRange(5.0, 100.0)
        self.spin_sync.setValue(30.0) 

        self.btn_export_csv = QPushButton("📊 Export Stats (CSV)")
        self.btn_export_csv.clicked.connect(self.export_stats_csv)
        self.btn_export_csv.setEnabled(False)

        self.btn_export_spikes = QPushButton("📥 Exporter Pics (Spike Trains)")
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
        layout_l3.addWidget(self.btn_export_spikes)
        layout_l3.addWidget(self.btn_export_img)

        main_layout.addLayout(layout_l1)
        main_layout.addLayout(layout_l2)
        main_layout.addLayout(layout_l3)

        # --- Bandeau Statistique ---
        self.stats_layout = QHBoxLayout()
        self.lbl_mu_count = QLabel("MUs Uniques Validées : <b>-</b>")
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
            
            # 2. CCA du Superviseur
            self.sources, w_x, self.autocor = CCAdecomp(filt_signal, self.spin_taux.value())
            
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
        
        fs = self.spin_fs.value() 
        candidates = []
        
        for i in range(self.sources.shape[0]):
            source = self.sources[i]
            
            # 3. K-MEANS (State of the Art) au lieu de l'ancien seuil manuel
            peaks, threshold = detect_spikes_kmeans(source, fs=fs)
            
            is_valid = False
            cov_isi = 1000 
            fr_hz = 0
            
            f, Pxx = welch(source, fs=fs, nperseg=1024)
            mu_spec = (np.sum(Pxx[(f >= 70) & (f <= 400)]) / np.sum(Pxx)) * 100 if np.sum(Pxx)>0 else 0
                
            if len(peaks) >= 4 and mu_spec >= self.spin_spec.value():
                isi = np.diff(peaks) 
                valid_isi = isi[(isi > 0.5 * np.median(isi)) & (isi < 1.5 * np.median(isi))]
                if len(valid_isi) >= 3:
                    cov_isi = (np.std(valid_isi) / np.mean(valid_isi)) * 100
                    if cov_isi <= self.spin_cov.value():
                        is_valid = True
                        fr_hz = 1.0 / (np.mean(valid_isi) / fs)
            
            candidates.append({"idx": i, "peaks": peaks, "is_valid": is_valid, "cov": cov_isi, "fr": fr_hz, "spec": mu_spec, "dup": False, "threshold": threshold})

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
                    if candidates[i]["cov"] > candidates[j]["cov"]: candidates[i]["dup"] = True
                    else: candidates[j]["dup"] = True

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
                
                # Ligne de seuil visuelle du K-Means
                line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('g', style=pg.QtCore.Qt.DashLine))
                line.setPos(threshold + (idx * self.source_offset))
                self.plot_sources.addItem(line)
                
                # ENREGISTREMENT POUR EXPORT
                nom_mu = f"{mu_count}"
                self.mu_spikes[nom_mu] = peaks
                
                # STATS
                self.mu_stats.append({
                    "MU": nom_mu, 
                    "Nb_Spikes": len(peaks), 
                    "Fr_Hz": cand["fr"], 
                    "CoV_%": cand["cov"],
                    "Correlation_CCA": self.autocor[idx]
                })
            else:
                self.plot_sources.plot(shifted, pen=pg.mkPen(color='#e0e0e0'))

        self.lbl_mu_count.setText(f"MUs Uniques Validées : <b>{mu_count}</b>")
        if mu_count > 0: 
            self.btn_export_csv.setEnabled(True)
            self.btn_export_spikes.setEnabled(True) 
            self.btn_export_img.setEnabled(True)

    def export_stats_csv(self):
        if not self.mu_stats: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Stats", "", "CSV (*.csv)")
        if file_path: pd.DataFrame(self.mu_stats).round(4).to_csv(file_path, index=False, sep=';', decimal=',')

    def export_spikes_csv(self):
        if not self.mu_spikes: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Exporter les Instants de Décharge", "", "Fichier CSV (*.csv)", options=options)
        
        if file_path:
            if not file_path.endswith('.csv'): file_path += '.csv'
            try:
                max_len = max([len(spk) for spk in self.mu_spikes.values()])
                export_dict = {}
                for mu_name, spk in self.mu_spikes.items():
                    padded_spk = np.pad(spk.astype(float), (0, max_len - len(spk)), constant_values=np.nan)
                    export_dict[mu_name] = padded_spk
                
                df_spikes = pd.DataFrame(export_dict)
                df_spikes.to_csv(file_path, index=False, sep=';')
                QMessageBox.information(self, "Succès", "Instants de décharge exportés avec succès !")
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