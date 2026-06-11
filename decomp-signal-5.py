import sys
import os
import site
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, welch

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

import pyqtgraph as pg
import pyqtgraph.exporters 
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QSpinBox, QLabel, QFileDialog, QMessageBox, QDoubleSpinBox)

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
    for out in outliers: sources_clean[out] = np.zeros_like(sources_clean[out])

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
        self.setWindowTitle("CCA Explorer - Décodage MUs & Déduplication")
        self.resize(1500, 950)
        
        self.signal = None
        self.sources = None
        self.source_offset = 0
        self.mu_stats = []
        self.mu_spikes = {} # NOUVEAU : Dictionnaire pour stocker les pics exportables
        
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- LIGNE 1 : Import & Traitement ---
        layout_l1 = QHBoxLayout()
        self.btn_load = QPushButton("📂 Charger fichier")
        self.btn_load.clicked.connect(self.open_file)
        
        self.btn_scca = QPushButton("🧹 Débruitage sCCA")
        self.btn_scca.clicked.connect(self.run_scca)
        self.btn_scca.setEnabled(False)

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
        self.spin_fs.setValue(2048) # Réglé sur ta norme Fair Benchmark
        self.spin_fs.setSingleStep(100)

        layout_l1.addWidget(self.btn_load)
        layout_l1.addWidget(self.btn_scca)
        layout_l1.addWidget(self.btn_run_cca)
        layout_l1.addWidget(lbl_taux)
        layout_l1.addWidget(self.spin_taux)
        layout_l1.addSpacing(20)
        layout_l1.addWidget(lbl_fs)
        layout_l1.addWidget(self.spin_fs)
        layout_l1.addStretch()

        # --- LIGNE 2 : Filtres Biologiques ---
        layout_l2 = QHBoxLayout()
        self.btn_spikes = QPushButton("🎯 Extraire & Dédupliquer MUs")
        self.btn_spikes.clicked.connect(self.run_spike_detection)
        self.btn_spikes.setEnabled(False)
        self.btn_spikes.setStyleSheet("background-color: #d1e7dd; font-weight: bold;")

        lbl_std = QLabel("Seuil Ampl. (xStd) :")
        self.spin_std = QDoubleSpinBox()
        self.spin_std.setRange(1.0, 10.0)
        self.spin_std.setValue(4.0)

        lbl_cov = QLabel("CoV Max (%) :")
        self.spin_cov = QDoubleSpinBox()
        self.spin_cov.setRange(5.0, 100.0)
        self.spin_cov.setValue(30.0)

        lbl_spec = QLabel("Énergie Spectrale Min (%) :")
        self.spin_spec = QDoubleSpinBox()
        self.spin_spec.setRange(0.0, 100.0)
        self.spin_spec.setValue(30.0)

        layout_l2.addWidget(self.btn_spikes)
        layout_l2.addWidget(lbl_std)
        layout_l2.addWidget(self.spin_std)
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

        # NOUVEAU BOUTON : EXPORT DES PICS
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
                self.btn_scca.setEnabled(True)
                self.btn_run_cca.setEnabled(True)
                self.plot_raw.clear()
                self.plot_sources.clear()
                offset = np.max(np.abs(self.signal)) * 1.5
                for i in range(self.signal.shape[0]): self.plot_raw.plot(self.signal[i] + (i * offset), pen='b')
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

    def run_scca(self):
        if self.signal is None: return
        fs = self.spin_fs.value()
        QApplication.setOverrideCursor(pg.QtCore.Qt.WaitCursor)
        try:
            clean_signal, outliers = sCCA_denoise(self.signal, fs=fs, f0=50.0, taux=self.spin_taux.value())
            self.signal = clean_signal 
            self.plot_raw.clear()
            offset = np.max(np.abs(self.signal)) * 1.5
            for i in range(self.signal.shape[0]): self.plot_raw.plot(self.signal[i] + (i * offset), pen='b')
            QMessageBox.information(self, "sCCA", f"{len(outliers)} source(s) 50Hz isolée(s).")
        finally:
            QApplication.restoreOverrideCursor()

    def run_cca(self):
        if self.signal is None: return
        self.sources, _, _ = CCAdecomp(self.signal, self.spin_taux.value())
        self.plot_sources.clear()
        self.source_offset = np.max(np.abs(self.sources)) * 1.5
        for i in range(self.sources.shape[0]):
            self.plot_sources.plot(self.sources[i] + (i * self.source_offset), pen=pg.mkPen(color='gray'))
        self.btn_spikes.setEnabled(True)

    def run_spike_detection(self):
        if self.sources is None: return
        self.plot_sources.clear()
        self.mu_stats = []
        self.mu_spikes = {} # Réinitialisation des pics
        
        fs = self.spin_fs.value() 
        candidates = []
        for i in range(self.sources.shape[0]):
            source = self.sources[i]
            peaks, _ = detect_spikes(source, std_multiplier=self.spin_std.value())
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
            
            candidates.append({"idx": i, "peaks": peaks, "is_valid": is_valid, "cov": cov_isi, "fr": fr_hz, "spec": mu_spec, "dup": False})

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
            idx, peaks = cand["idx"], cand["peaks"]
            shifted = self.sources[idx] + (idx * self.source_offset)
            
            if cand["is_valid"] and not cand["dup"]:
                mu_count += 1
                color = colors[mu_count % len(colors)]
                self.plot_sources.plot(shifted, pen=pg.mkPen(color=color, width=1.5))
                self.plot_sources.addItem(pg.ScatterPlotItem(x=peaks, y=shifted[peaks], size=8, pen=None, brush=pg.mkBrush('r')))
                
                # ENREGISTREMENT DU PIC POUR L'EXPORT
                nom_mu = f"CCA_MU_{mu_count}"
                self.mu_spikes[nom_mu] = peaks

                self.mu_stats.append({"MU": nom_mu, "Nb_Spikes": len(peaks), "Fr_Hz": cand["fr"], "CoV_%": cand["cov"]})
            else:
                self.plot_sources.plot(shifted, pen=pg.mkPen(color='#e0e0e0'))

        self.lbl_mu_count.setText(f"MUs Uniques Validées : <b>{mu_count}</b>")
        if mu_count > 0: 
            self.btn_export_csv.setEnabled(True)
            self.btn_export_spikes.setEnabled(True) # Activer le nouveau bouton
            self.btn_export_img.setEnabled(True)

    def export_stats_csv(self):
        if not self.mu_stats: return
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Stats", "", "CSV (*.csv)")
        if file_path: pd.DataFrame(self.mu_stats).round(2).to_csv(file_path, index=False, sep=';', decimal=',')

    # NOUVELLE FONCTION D'EXPORT DES PICS (Spike Trains)
    def export_spikes_csv(self):
        if not self.mu_spikes: return
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Exporter les Instants de Décharge", "", "Fichier CSV (*.csv)", options=options)
        
        if file_path:
            if not file_path.endswith('.csv'): file_path += '.csv'
            try:
                # On doit égaliser la longueur des colonnes avec des 'NaN' (cases vides) car chaque MU a un nombre différent de pics
                max_len = max([len(spk) for spk in self.mu_spikes.values()])
                export_dict = {}
                
                for mu_name, spk in self.mu_spikes.items():
                    padded_spk = np.pad(spk.astype(float), (0, max_len - len(spk)), constant_values=np.nan)
                    export_dict[mu_name] = padded_spk
                
                # Création et sauvegarde du DataFrame
                df_spikes = pd.DataFrame(export_dict)
                df_spikes.to_csv(file_path, index=False, sep=';')
                
                QMessageBox.information(self, "Succès", f"Instants de décharge exportés avec succès !\n\nChaque colonne représente une Unité Motrice.\nLes valeurs sont les index (échantillons) temporels.")
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