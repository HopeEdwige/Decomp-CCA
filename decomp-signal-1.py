import sys
import os
import site
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat
from scipy.signal import find_peaks, welch, iirnotch, filtfilt

if sys.platform.startswith('win'):
    qt_paths = []
    if hasattr(site, 'getsitepackages'):
        qt_paths.extend(site.getsitepackages())
    qt_paths.append(site.getusersitepackages())
    qt_paths.extend(sys.path)
    for base in qt_paths:
        if not base:
            continue
        qt_bin = os.path.join(base, 'PyQt5', 'Qt5', 'bin')
        qt_plugins = os.path.join(base, 'PyQt5', 'Qt5', 'plugins')
        if os.path.isdir(qt_bin):
            try:
                os.add_dll_directory(qt_bin)
            except AttributeError:
                pass
            os.environ['PATH'] = qt_bin + os.pathsep + os.environ.get('PATH', '')
        if os.path.isdir(qt_plugins):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugins
        if os.path.isdir(qt_bin) or os.path.isdir(qt_plugins):
            break

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QSpinBox, QLabel, QFileDialog, QMessageBox, QDoubleSpinBox)
import pyqtgraph as pg

# --- 1. Algorithmes Mathématiques ---

def CCAdecomp(sig, taux):
    sig = sig - sig.mean(axis=1, keepdims=True)
    x = sig[:, :-taux]
    y = sig[:, taux:]
    
    Q_x, R_x = np.linalg.qr(x.T, mode='reduced')
    Q_y, R_y = np.linalg.qr(y.T, mode='reduced')
    
    U, S, Vt = np.linalg.svd(Q_x.T @ Q_y, full_matrices=False)
    sources = (Q_x @ U).T
    w_x = np.linalg.solve(R_x, U)
    return sources, w_x, S

def sCCA_denoise(sig, fs=2048, f0=50.0, taux=1):
    sig_mean = sig.mean(axis=1, keepdims=True)
    x_c = sig - sig_mean
    x_d = x_c[:, :-taux]
    y_d = x_c[:, taux:]

    Q_x, R_x = np.linalg.qr(x_d.T, mode='reduced')
    Q_y, R_y = np.linalg.qr(y_d.T, mode='reduced')
    U, S, Vt = np.linalg.svd(Q_x.T @ Q_y, full_matrices=False)
    
    sources = (Q_x @ U).T
    w_x = np.linalg.solve(R_x, U)
    A = np.linalg.pinv(w_x)

    pli_powers = []
    for i in range(sources.shape[0]):
        f, Pxx = welch(sources[i], fs=fs, nperseg=1024)
        idx_50 = np.argmin(np.abs(f - f0))
        power_50 = np.sum(Pxx[max(0, idx_50-1) : min(len(Pxx), idx_50+2)])
        pli_powers.append(power_50)

    pli_powers = np.array(pli_powers)
    Q1 = np.percentile(pli_powers, 25)
    Q3 = np.percentile(pli_powers, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    outliers = np.where(pli_powers > upper_bound)[0]

    sources_clean = sources.copy()
    
    # Méthode d'annulation (Zeroing) pour éviter les distorsions de phase
    for out in outliers:
        sources_clean[out] = np.zeros_like(sources_clean[out])

    sig_recon_d = A @ sources_clean
    final_sig = np.zeros_like(sig)
    final_sig[:, :-taux] = sig_recon_d
    final_sig[:, -taux:] = x_c[:, -taux:] 
    final_sig = final_sig + sig_mean
    return final_sig, outliers

def detect_spikes(source, std_multiplier=4.0, min_distance=20):
    signal_abs = np.abs(source)
    threshold = np.mean(signal_abs) + (std_multiplier * np.std(signal_abs))
    peaks, _ = find_peaks(signal_abs, height=threshold, distance=min_distance)
    return peaks, threshold


# --- 2. Interface Graphique Principale ---

class CCAMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCA Explorer - Décodage MUs & Déduplication (Rectus Femoris)")
        self.resize(1500, 950)
        
        self.signal = None
        self.sources = None
        self.source_offset = 0
        self.mu_stats = []
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- LIGNE 1 : Import & Traitement lourd ---
        layout_l1 = QHBoxLayout()
        self.btn_load = QPushButton("📂 Charger fichier")
        self.btn_load.clicked.connect(self.open_file)
        
        self.btn_scca = QPushButton("🧹 Débruitage sCCA (Anti-50Hz)")
        self.btn_scca.clicked.connect(self.run_scca)
        self.btn_scca.setEnabled(False)
        self.btn_scca.setStyleSheet("background-color: #cfe2ff; font-weight: bold;")

        self.btn_run_cca = QPushButton("⚡ Décomposition CCA")
        self.btn_run_cca.clicked.connect(self.run_cca)
        self.btn_run_cca.setEnabled(False)
        
        lbl_taux = QLabel("Taux CCA :")
        self.spin_taux = QSpinBox()
        self.spin_taux.setRange(1, 500)
        self.spin_taux.setValue(1)

        lbl_fs = QLabel("Fréq. Échantillonnage (Hz) :")
        self.spin_fs = QSpinBox()
        self.spin_fs.setRange(100, 20000)
        self.spin_fs.setValue(2048) # AJUSTÉ PAR DÉFAUT POUR TA BASE SYNCHRO (10kHz)
        self.spin_fs.setSingleStep(1000)

        layout_l1.addWidget(self.btn_load)
        layout_l1.addWidget(self.btn_scca)
        layout_l1.addWidget(self.btn_run_cca)
        layout_l1.addWidget(lbl_taux)
        layout_l1.addWidget(self.spin_taux)
        layout_l1.addSpacing(20)
        layout_l1.addWidget(lbl_fs)
        layout_l1.addWidget(self.spin_fs)
        layout_l1.addStretch()

        # --- LIGNE 2 : Filtres de Détection (Biologiques) ---
        layout_l2 = QHBoxLayout()
        self.btn_spikes = QPushButton("🎯 Extraire & Dédupliquer MUs")
        self.btn_spikes.clicked.connect(self.run_spike_detection)
        self.btn_spikes.setEnabled(False)
        self.btn_spikes.setStyleSheet("background-color: #d1e7dd; font-weight: bold;")

        lbl_std = QLabel("Seuil Ampl. (xStd) :")
        self.spin_std = QDoubleSpinBox()
        self.spin_std.setRange(1.0, 10.0)
        self.spin_std.setValue(4.0) # Standard optimal
        self.spin_std.setSingleStep(0.5)

        lbl_cov = QLabel("CoV Max (%) :")
        self.spin_cov = QDoubleSpinBox()
        self.spin_cov.setRange(5.0, 100.0)
        self.spin_cov.setValue(30.0) # Standard physiologique activé par défaut
        self.spin_cov.setSingleStep(5.0)

        lbl_spec = QLabel("Énergie Spectrale Min (%) :")
        self.spin_spec = QDoubleSpinBox()
        self.spin_spec.setRange(0.0, 100.0)
        self.spin_spec.setValue(30.0) # Bande EMG active par défaut
        self.spin_spec.setSingleStep(5.0)

        layout_l2.addWidget(self.btn_spikes)
        layout_l2.addWidget(lbl_std)
        layout_l2.addWidget(self.spin_std)
        layout_l2.addSpacing(15)
        layout_l2.addWidget(lbl_cov)
        layout_l2.addWidget(self.spin_cov)
        layout_l2.addSpacing(15)
        layout_l2.addWidget(lbl_spec)
        layout_l2.addWidget(self.spin_spec)
        layout_l2.addStretch()

        # --- LIGNE 3 : Déduplication et Export ---
        layout_l3 = QHBoxLayout()
        
        lbl_sync = QLabel("Taux de Coïncidence Max (Déduplication %) :")
        lbl_sync.setStyleSheet("color: #856404; font-weight: bold;")
        self.spin_sync = QDoubleSpinBox()
        self.spin_sync.setRange(5.0, 100.0)
        self.spin_sync.setValue(30.0) 
        self.spin_sync.setSingleStep(5.0)

        self.btn_export_csv = QPushButton("📊 Export CSV (Stats)")
        self.btn_export_csv.clicked.connect(self.export_stats_csv)
        self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.setStyleSheet("background-color: #fff3cd;")

        self.btn_export_img = QPushButton("📸 Export Image")
        self.btn_export_img.clicked.connect(self.export_as_image)
        self.btn_export_img.setEnabled(False)
        self.btn_export_img.setStyleSheet("background-color: #cff4fc;")

        layout_l3.addWidget(lbl_sync)
        layout_l3.addWidget(self.spin_sync)
        layout_l3.addStretch()
        layout_l3.addWidget(self.btn_export_csv)
        layout_l3.addWidget(self.btn_export_img)

        main_layout.addLayout(layout_l1)
        main_layout.addLayout(layout_l2)
        main_layout.addLayout(layout_l3)

        # --- Bandeau Statistique ---
        self.stats_layout = QHBoxLayout()
        self.lbl_mu_count = QLabel("MUs Uniques Validées : <b>-</b>")
        self.lbl_mu_count.setStyleSheet("font-size: 13pt; color: #0f5132; padding: 5px; background-color: #e2f0d9; border-radius: 4px;")
        self.stats_layout.addWidget(self.lbl_mu_count)
        self.stats_layout.addStretch()
        main_layout.addLayout(self.stats_layout)

        # --- Graphiques ---
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.plot_raw = pg.PlotWidget(title="Signaux Importés (ou Nettoyés sCCA)")
        main_layout.addWidget(self.plot_raw)

        self.plot_sources = pg.PlotWidget(title="Sources CCA (Rouge = Vraie MU Unique, Gris = Bruit ou Doublon rejeté)")
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
                self.btn_scca.setEnabled(True)
                self.btn_run_cca.setEnabled(True)
                self.btn_spikes.setEnabled(False)
                self.btn_export_img.setEnabled(False)
                self.btn_export_csv.setEnabled(False)
                self.sources = None
                self.lbl_mu_count.setText("MUs Uniques Validées : <b>-</b>")
                self.plot_raw_signals()
                self.plot_sources.clear()
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier :\n{str(e)}")

    def plot_raw_signals(self):
        self.plot_raw.clear()
        if self.signal is None: return
        offset = np.max(np.abs(self.signal)) * 1.5
        for i in range(self.signal.shape[0]): 
            self.plot_raw.plot(self.signal[i] + (i * offset), pen=pg.mkPen('b', width=1))

    def run_scca(self):
        if self.signal is None: return
        fs = self.spin_fs.value()
        QApplication.setOverrideCursor(pg.QtCore.Qt.WaitCursor)
        try:
            clean_signal, outliers = sCCA_denoise(self.signal, fs=fs, f0=50.0, taux=self.spin_taux.value())
            self.signal = clean_signal 
            self.plot_raw_signals()    
            QMessageBox.information(self, "sCCA Terminé", f"{len(outliers)} source(s) polluée(s) par le 50Hz isolée(s) et annulée(s) via IQR.")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def run_cca(self):
        if self.signal is None: return
        self.sources, _, _ = CCAdecomp(self.signal, self.spin_taux.value())
        self.plot_sources.clear()
        self.source_offset = np.max(np.abs(self.sources)) * 1.5
        colors = ['#D95319', '#EDB120', '#7E2F8E', '#77AC30']
        for i in range(self.sources.shape[0]):
            color = colors[i % len(colors)]
            shifted_source = self.sources[i] + (i * self.source_offset)
            self.plot_sources.plot(shifted_source, pen=pg.mkPen(color=color, width=1.2))
        self.btn_spikes.setEnabled(True)
        self.btn_export_img.setEnabled(True)

    def run_spike_detection(self):
        if self.sources is None: return
        self.plot_sources.clear()
        self.mu_stats = [] 
        
        fs = self.spin_fs.value() 
        std_multiplier = self.spin_std.value()
        cov_threshold = self.spin_cov.value()
        spec_threshold = self.spin_spec.value() 
        sync_threshold = self.spin_sync.value() / 100.0 
        
        candidates = []
        colors = ['#D95319', '#EDB120', '#7E2F8E', '#77AC30']
        
        for i in range(self.sources.shape[0]):
            source = self.sources[i]
            peaks, _ = detect_spikes(source, std_multiplier=std_multiplier)
            
            is_valid_candidate = False
            cov_isi = 1000 
            firing_rate_hz = 0
            mu_spectral_ratio = 0
            
            f, Pxx = welch(source, fs=fs, nperseg=1024)
            mu_band_power = np.sum(Pxx[(f >= 70) & (f <= 400)])
            total_power = np.sum(Pxx)
            if total_power > 0:
                mu_spectral_ratio = (mu_band_power / total_power) * 100
                
            if len(peaks) >= 4 and mu_spectral_ratio >= spec_threshold:
                isi_samples = np.diff(peaks) 
                median_isi = np.median(isi_samples)
                valid_isi_samples = isi_samples[(isi_samples > 0.5 * median_isi) & (isi_samples < 1.5 * median_isi)]
                
                if len(valid_isi_samples) >= 3:
                    cov_isi = (np.std(valid_isi_samples) / np.mean(valid_isi_samples)) * 100
                    if cov_isi <= cov_threshold:
                        is_valid_candidate = True
                        mean_isi_sec = np.mean(valid_isi_samples) / fs
                        firing_rate_hz = 1.0 / mean_isi_sec
            
            candidates.append({
                "idx": i,
                "peaks": peaks,
                "is_valid": is_valid_candidate,
                "cov": cov_isi,
                "fr_hz": firing_rate_hz,
                "spec_ratio": mu_spectral_ratio,
                "is_duplicate": False 
            })

        tolerance_samples = int((fs / 1000.0) * 1.5) 
        
        for i in range(len(candidates)):
            if not candidates[i]["is_valid"] or candidates[i]["is_duplicate"]: continue
            for j in range(i + 1, len(candidates)):
                if not candidates[j]["is_valid"] or candidates[j]["is_duplicate"]: continue
                
                peaks_i = candidates[i]["peaks"]
                peaks_j = candidates[j]["peaks"]
                
                common_spikes = 0
                for p_i in peaks_i:
                    if np.any(np.abs(peaks_j - p_i) <= tolerance_samples):
                        common_spikes += 1
                
                max_ratio = common_spikes / min(len(peaks_i), len(peaks_j))
                if max_ratio > sync_threshold:
                    if candidates[i]["cov"] > candidates[j]["cov"]:
                        candidates[i]["is_duplicate"] = True
                        break 
                    else:
                        candidates[j]["is_duplicate"] = True

        mu_detected_count = 0
        for cand in candidates:
            idx = cand["idx"]
            color = colors[idx % len(colors)]
            shifted_source = self.sources[idx] + (idx * self.source_offset)
            self.plot_sources.plot(shifted_source, pen=pg.mkPen(color=color, width=1.2))
            peaks = cand["peaks"]
            
            if cand["is_valid"] and not cand["is_duplicate"]:
                mu_detected_count += 1
                self.mu_stats.append({
                            "MU_Index": idx + 1,
                            "Nb_Spikes": len(peaks),
                            "Firing_Rate_Hz": cand["fr_hz"],
                            "CoV_ISI_%": cand["cov"],
                            "Energie_MU_%": cand["spec_ratio"]
                        })
                scatter = pg.ScatterPlotItem(x=peaks, y=shifted_source[peaks], size=8, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 200))
            elif len(peaks) > 0:
                scatter = pg.ScatterPlotItem(x=peaks, y=shifted_source[peaks], size=5, pen=pg.mkPen(None), brush=pg.mkBrush(150, 150, 150, 150))
            
            if len(peaks) > 0:
                self.plot_sources.addItem(scatter)
        
        self.lbl_mu_count.setText(f"MUs Uniques Validées : <b>{mu_detected_count}</b>")
        if mu_detected_count > 0: self.btn_export_csv.setEnabled(True) 
        QMessageBox.information(self, "Analyse Terminée", f"Traitement complet terminé.\nIl reste {mu_detected_count} MUs uniques et biologiquement valides.")

    def export_stats_csv(self):
        if not self.mu_stats: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer les Statistiques (Excel)", "", "Fichier CSV (*.csv)", options=options)
        if file_path:
            if not file_path.endswith('.csv'): file_path += '.csv'
            try:
                df = pd.DataFrame(self.mu_stats)
                df = df.round(2) 
                df.to_csv(file_path, index=False, sep=';', decimal=',') 
                QMessageBox.information(self, "Succès", f"Statistiques exportées avec succès.")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", str(e))

    def export_as_image(self):
        if self.sources is None: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le graphique", "", "Images PNG (*.png)", options=options)
        if file_path:
            if not file_path.endswith('.png'): file_path += '.png'
            try:
                exporter = pg.exporters.ImageExporter(self.plot_sources.plotItem)
                exporter.parameters()['width'] = 2500 
                exporter.export(file_path)
                QMessageBox.information(self, "Succès", "Graphique exporté avec succès.")
            except Exception as e:
                QMessageBox.critical(self, "Erreur", str(e))

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    window = CCAMainWindow()
    window.show()
    sys.exit(app.exec_())