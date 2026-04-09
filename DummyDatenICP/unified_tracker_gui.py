import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import h5py
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import time


class PointCloudAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("H5 Point Cloud Fast Viewer & Tracker")
        self.geometry("750x850")

        # --- Datenstruktur ---
        self.h5_path = ""
        self.frames = []  # Liste der Punktwolken-Daten (N, 3)
        self.timestamps = []  # Zeitstempel pro Frame
        self.pcds = []  # Open3D PointCloud Objekte
        self.current_frame = 0

        self.vis = None  # Open3D Visualizer Instanz
        self.pcd_vis = None  # Aktuelle Punktwolke im Visualizer

        self.setup_ui()
        self.poll_o3d()  # Hält das Open3D Fenster parallel zu Tkinter am Leben

    def setup_ui(self):
        # --- 1. Load Data ---
        frame1 = ttk.LabelFrame(self, text="1. Daten laden (H5)")
        frame1.pack(fill="x", padx=10, pady=5)

        ttk.Button(frame1, text="H5 Datei laden", command=self.load_data).pack(side="left", padx=10, pady=10)
        self.lbl_file = ttk.Label(frame1, text="Keine Datei geladen.")
        self.lbl_file.pack(side="left", padx=10, pady=10)

        # --- 2. Fast Viewer & Diagnose ---
        frame2 = ttk.LabelFrame(self, text="2. Fast Viewer & Diagnose")
        frame2.pack(fill="x", padx=10, pady=5)

        self.slider = ttk.Scale(frame2, from_=0, to=0, orient="horizontal", command=self.on_slider_change)
        self.slider.pack(fill="x", padx=10, pady=5)

        self.lbl_frame_info = ttk.Label(frame2, text="Frame: 0 | Timestamp: -")
        self.lbl_frame_info.pack(pady=5)

        btn_frame2 = ttk.Frame(frame2)
        btn_frame2.pack(fill="x")
        ttk.Button(btn_frame2, text="Open3D Viewer öffnen", command=self.start_viewer).pack(side="left", padx=10,
                                                                                            pady=5)
        ttk.Button(btn_frame2, text="H5 Diagnose (Struktur) anzeigen", command=self.show_diagnosis).pack(side="left",
                                                                                                         padx=10,
                                                                                                         pady=5)

        # --- 3. ROI (Region of Interest) ---
        frame3 = ttk.LabelFrame(self, text="3. ROI Bereinigung")
        frame3.pack(fill="x", padx=10, pady=5)

        self.roi_vars = {}
        roi_grid = ttk.Frame(frame3)
        roi_grid.pack(pady=5)

        for i, axis in enumerate(['X', 'Y', 'Z']):
            ttk.Label(roi_grid, text=f"{axis} Min:").grid(row=i, column=0, padx=5, pady=2)
            var_min = tk.DoubleVar(value=-500.0)
            ttk.Entry(roi_grid, textvariable=var_min, width=10).grid(row=i, column=1, padx=5, pady=2)

            ttk.Label(roi_grid, text=f"{axis} Max:").grid(row=i, column=2, padx=5, pady=2)
            var_max = tk.DoubleVar(value=500.0)
            ttk.Entry(roi_grid, textvariable=var_max, width=10).grid(row=i, column=3, padx=5, pady=2)

            self.roi_vars[f"{axis.lower()}_min"] = var_min
            self.roi_vars[f"{axis.lower()}_max"] = var_max

        ttk.Button(frame3, text="ROI auf aktuellen Frame anwenden", command=self.apply_roi_current).pack(pady=5)

        # --- 4. DBSCAN ---
        frame4 = ttk.LabelFrame(self, text="4. DBSCAN (Phantomwolke extrahieren)")
        frame4.pack(fill="x", padx=10, pady=5)

        ttk.Button(frame4, text="DBSCAN auf alle Frames anwenden", command=self.run_dbscan).pack(pady=10)

        # --- 5. Tracking ---
        frame5 = ttk.LabelFrame(self, text="5. ICP Tracking")
        frame5.pack(fill="x", padx=10, pady=5)

        track_grid = ttk.Frame(frame5)
        track_grid.pack(pady=5)

        ttk.Label(track_grid, text="Start Frame (leer = 0):").grid(row=0, column=0, padx=5)
        self.var_start_frame = tk.StringVar()
        ttk.Entry(track_grid, textvariable=self.var_start_frame, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(track_grid, text="End Frame (leer = Ende):").grid(row=0, column=2, padx=5)
        self.var_end_frame = tk.StringVar()
        ttk.Entry(track_grid, textvariable=self.var_end_frame, width=8).grid(row=0, column=3, padx=5)

        ttk.Button(frame5, text="Start Tracking", command=self.start_tracking).pack(pady=10)

        # --- 6. Tracking Analyse (Das Verrückte Feature) ---
        frame6 = ttk.LabelFrame(self, text="🚀 Analyse Tracking CSV")
        frame6.pack(fill="x", padx=10, pady=5)
        ttk.Button(frame6, text="CSV Ergebnisse Plotten", command=self.plot_tracking_results).pack(pady=10)

        # --- Progress Bar ---
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=10)
        self.lbl_status = ttk.Label(self, text="Status: Bereit")
        self.lbl_status.pack(pady=5)

    # ---------------------------------------------------------
    # 1. & 2. Daten laden und H5 Viewer
    # ---------------------------------------------------------
    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5")])
        if not path: return
        self.h5_path = path
        self.lbl_file.config(text=os.path.basename(path))

        self.lbl_status.config(text="Status: Lade H5 Daten...")
        self.update()

        try:
            with h5py.File(path, 'r') as f:
                # Annahme: Daten liegen strukturiert vor (Anpassen an deine H5-Struktur)
                # Suchen nach Keys, die wie Frames aussehen, oder iterieren
                keys = list(f.keys())
                self.frames = []
                self.timestamps = []
                self.pcds = []

                # Sehr rudimentäre H5 Extraktion (muss je nach File evt. angepasst werden)
                for i, k in enumerate(sorted(keys)):
                    data = np.array(f[k])
                    if data.shape[1] >= 3:
                        points = data[:, :3]  # Nimm nur X,Y,Z
                        self.frames.append(points)
                        self.timestamps.append(str(k))  # Key als Timestamp

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points)
                        self.pcds.append(pcd)

            self.slider.config(to=len(self.frames) - 1)
            self.lbl_status.config(text=f"Status: {len(self.frames)} Frames geladen.")
            self.on_slider_change(0)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte H5 nicht laden:\n{e}")

    def show_diagnosis(self):
        if not self.h5_path: return
        diag_win = tk.Toplevel(self)
        diag_win.title("H5 Diagnose")
        diag_win.geometry("400x500")
        text = tk.Text(diag_win, wrap="word")
        text.pack(expand=True, fill="both")

        def print_struct(name, obj):
            text.insert("end", f"{name} : {type(obj)}\n")

        with h5py.File(self.h5_path, 'r') as f:
            f.visititems(print_struct)

    def on_slider_change(self, val):
        if not self.frames: return
        self.current_frame = int(float(self.slider.get()))
        ts = self.timestamps[self.current_frame]
        self.lbl_frame_info.config(text=f"Frame: {self.current_frame} | Timestamp: {ts}")
        self.update_viewer_geometry()

    def start_viewer(self):
        if not self.frames: return
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name="H5 Fast Viewer")
            self.pcd_vis = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd_vis)
        self.update_viewer_geometry()

    def update_viewer_geometry(self):
        if self.vis and self.frames:
            self.pcd_vis.points = self.pcds[self.current_frame].points
            self.pcd_vis.colors = self.pcds[self.current_frame].colors
            self.vis.update_geometry(self.pcd_vis)
            self.vis.poll_events()
            self.vis.update_renderer()

    def poll_o3d(self):
        """Hält das Open3D Fenster responsiv, ohne Tkinter zu blockieren."""
        if self.vis is not None:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.after(50, self.poll_o3d)

    # ---------------------------------------------------------
    # 3. ROI
    # ---------------------------------------------------------
    def apply_roi_current(self):
        if not self.frames: return
        x_min, x_max = self.roi_vars['x_min'].get(), self.roi_vars['x_max'].get()
        y_min, y_max = self.roi_vars['y_min'].get(), self.roi_vars['y_max'].get()
        z_min, z_max = self.roi_vars['z_min'].get(), self.roi_vars['z_max'].get()

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(x_min, y_min, z_min),
            max_bound=(x_max, y_max, z_max)
        )

        # Test an aktuellem Frame
        cropped_pcd = self.pcds[self.current_frame].crop(bbox)
        self.pcds[self.current_frame] = cropped_pcd
        self.update_viewer_geometry()
        self.lbl_status.config(text="Status: ROI auf aktuellen Frame angewendet.")

    # ---------------------------------------------------------
    # 4. DBSCAN
    # ---------------------------------------------------------
    def run_dbscan(self):
        if not self.frames: return

        def task():
            self.lbl_status.config(text="Status: Führe DBSCAN aus (Fenster ggf. kurz eingefroren)...")
            self.progress["maximum"] = len(self.pcds)

            for i, pcd in enumerate(self.pcds):
                # DBSCAN Parameter ggf. an deine Werte anpassen
                labels = np.array(pcd.cluster_dbscan(eps=20.0, min_points=10, print_progress=False))
                if len(labels) == 0: continue

                # Finde das größte Cluster (Phantomwolke)
                max_label = labels.max()
                if max_label >= 0:
                    counts = np.bincount(labels[labels >= 0])
                    phantom_label = np.argmax(counts)

                    # Behalte nur Indizes des Phantom-Clusters
                    phantom_indices = np.where(labels == phantom_label)[0]
                    self.pcds[i] = pcd.select_by_index(phantom_indices)

                self.progress["value"] = i + 1
                self.update_idletasks()

            self.lbl_status.config(text="Status: DBSCAN abgeschlossen.")
            self.update_viewer_geometry()

        threading.Thread(target=task, daemon=True).start()

    # ---------------------------------------------------------
    # 5. Tracking
    # ---------------------------------------------------------
    def start_tracking(self):
        if not self.frames: return

        start_str = self.var_start_frame.get().strip()
        end_str = self.var_end_frame.get().strip()

        start_idx = int(start_str) if start_str else 0
        end_idx = int(end_str) if end_str else len(self.pcds) - 1

        # Speicherort abfragen
        default_name = f"ICP_{os.path.basename(self.h5_path)}_{int(time.time())}.csv"
        save_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not save_path: return

        def task():
            self.lbl_status.config(text=f"Status: Tracking von Frame {start_idx} bis {end_idx}...")
            self.progress["maximum"] = end_idx - start_idx

            results = []
            target = self.pcds[start_idx]  # initiales Target

            for i in range(start_idx, end_idx + 1):
                source = self.pcds[i]
                ts = self.timestamps[i]

                # --- HIER TRACKING_V2.PY LOGIK EINSETZEN ---
                # Placeholder ICP:
                trans_init = np.eye(4)
                threshold = 50.0
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source, target, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )

                # Extrahiere Translation & Rotation
                T = reg_p2p.transformation
                tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
                # Einfache Näherung für Euler-Winkel aus Rotationsmatrix für den Placeholder
                rx, ry, rz = np.arctan2(T[2, 1], T[2, 2]), np.arctan2(-T[2, 0],
                                                                      np.sqrt(T[2, 1] ** 2 + T[2, 2] ** 2)), np.arctan2(
                    T[1, 0], T[0, 0])

                rmse = reg_p2p.inlier_rmse
                score = reg_p2p.fitness * 100

                results.append([ts, tx, ty, tz, rx, ry, rz, score, rmse])

                self.progress["value"] = i - start_idx + 1
                self.update_idletasks()

            # Speichern
            df = pd.DataFrame(results,
                              columns=["Timestamp (steady)", "Current Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "Score",
                                       "RMSE3D"])
            df.to_csv(save_path, index=False)
            self.lbl_status.config(text=f"Status: Tracking beendet. Gespeichert in {os.path.basename(save_path)}")

        threading.Thread(target=task, daemon=True).start()

    # ---------------------------------------------------------
    # 6. Tracking Analyse (CSV Plotter)
    # ---------------------------------------------------------
    def plot_tracking_results(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not csv_path: return

        df = pd.read_csv(csv_path)

        # Neues Fenster
        plot_win = tk.Toplevel(self)
        plot_win.title(f"Tracking Analyse - {os.path.basename(csv_path)}")
        plot_win.geometry("900x700")

        # Checkboxen für Sichtbarkeiten
        controls = ttk.Frame(plot_win)
        controls.pack(fill="x", padx=10, pady=5)

        show_trans = tk.BooleanVar(value=True)
        show_rot = tk.BooleanVar(value=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        canvas = FigureCanvasTkAgg(fig, master=plot_win)

        def update_plot():
            ax1.clear()
            ax2.clear()

            x_vals = range(len(df))  # Oder pd.to_datetime wenn steady timestamp parsebar ist

            if show_trans.get():
                if "Current Tx" in df.columns:
                    ax1.plot(x_vals, df["Current Tx"], label="Tx", linestyle='-')
                    ax1.plot(x_vals, df["Ty"], label="Ty", linestyle='-')
                    ax1.plot(x_vals, df["Tz"], label="Tz", linestyle='-')
            if show_rot.get():
                if "Rx" in df.columns:
                    # Rotations auf zweiter Achse oder gestrichelt
                    ax1.plot(x_vals, df["Rx"], label="Rx", linestyle='--')
                    ax1.plot(x_vals, df["Ry"], label="Ry", linestyle='--')
                    ax1.plot(x_vals, df["Rz"], label="Rz", linestyle='--')

            ax1.set_title("6 Degrees of Freedom über die Zeit")
            ax1.set_ylabel("Translation / Rotation")
            ax1.legend(loc="upper right")
            ax1.grid(True)

            if "RMSE3D" in df.columns:
                ax2.plot(x_vals, df["RMSE3D"], color='red', label="RMSE 3D")
                ax2.set_title("Fehler (RMSE)")
                ax2.set_ylabel("RMSE")
                ax2.set_xlabel("Frames")
                ax2.legend()
                ax2.grid(True)

            fig.tight_layout()
            canvas.draw()

        ttk.Checkbutton(controls, text="Translation zeigen (Tx, Ty, Tz)", variable=show_trans,
                        command=update_plot).pack(side="left", padx=10)
        ttk.Checkbutton(controls, text="Rotation zeigen (Rx, Ry, Rz)", variable=show_rot, command=update_plot).pack(
            side="left", padx=10)

        canvas.get_tk_widget().pack(fill="both", expand=True)
        update_plot()


if __name__ == "__main__":
    app = PointCloudAnalyzerApp()
    app.mainloop()