import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import time
import json
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# --- Import deines eigenen Loaders ---
try:
    from extractH5.h5_loader import H5PointCloudStream
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent / "DummyDatenICP"))
    from extractH5.h5_loader import H5PointCloudStream


class PointCloudAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("H5 Point Cloud Fast Viewer & Tracker")
        self.geometry("750x850")

        # --- Datenstruktur ---
        self.h5_path = ""
        self.stream = None
        self.frames = []
        self.timestamps = []
        self.pcds = []
        self.current_frame = 0

        self.vis = None
        self.pcd_vis = None

        # NEU: Variablen für die ROI Bounding Box
        self.roi_bbox = None  # Speichert die mathematische Box
        self.bbox_vis = None  # Speichert das visuelle schwarze Gitter im Viewer

        self.setup_ui()
        self.poll_o3d()

    def setup_ui(self):
        # --- 1. Load Data ---
        frame1 = ttk.LabelFrame(self, text="1. Daten laden (H5)")
        frame1.pack(fill="x", padx=10, pady=5)

        ttk.Button(frame1, text="H5 Datei laden", command=self.load_data).pack(side="left", padx=10, pady=10)
        self.lbl_file = ttk.Label(frame1, text="Keine Datei geladen.")
        self.lbl_file.pack(side="left", padx=10, pady=10)

        # --- 2. Fast Viewer ---
        frame2 = ttk.LabelFrame(self, text="2. Fast Viewer")
        frame2.pack(fill="x", padx=10, pady=5)

        self.slider = ttk.Scale(frame2, from_=0, to=0, orient="horizontal", command=self.on_slider_change)
        self.slider.pack(fill="x", padx=10, pady=5)

        self.lbl_frame_info = ttk.Label(frame2, text="Frame: 0 | Timestamp: -")
        self.lbl_frame_info.pack(pady=5)

        btn_frame2 = ttk.Frame(frame2)
        btn_frame2.pack(fill="x")
        ttk.Button(btn_frame2, text="Open3D Viewer öffnen", command=self.start_viewer).pack(side="left", padx=10,
                                                                                            pady=5)

        # --- 3. ROI (Region of Interest) ---
        frame3 = ttk.LabelFrame(self, text="3. ROI Visualisierung")
        frame3.pack(fill="x", padx=10, pady=5)

        self.roi_vars = {}
        roi_grid = ttk.Frame(frame3)
        roi_grid.pack(pady=5)

        for i, axis in enumerate(['X', 'Y', 'Z']):
            ttk.Label(roi_grid, text=f"{axis} Min:").grid(row=i, column=0, padx=5, pady=2)
            var_min = tk.DoubleVar(value=-200.0)
            ttk.Entry(roi_grid, textvariable=var_min, width=10).grid(row=i, column=1, padx=5, pady=2)

            ttk.Label(roi_grid, text=f"{axis} Max:").grid(row=i, column=2, padx=5, pady=2)
            var_max = tk.DoubleVar(value=200.0)
            ttk.Entry(roi_grid, textvariable=var_max, width=10).grid(row=i, column=3, padx=5, pady=2)

            self.roi_vars[f"{axis.lower()}_min"] = var_min
            self.roi_vars[f"{axis.lower()}_max"] = var_max

        ttk.Button(frame3, text="ROI als Box einblenden (Vorschau)", command=self.apply_roi_current).pack(pady=5)
        ttk.Button(frame3, text="ROI als JSON speichern", command=self.save_roi).pack(pady=5)
        # --- 4. DBSCAN ---
        frame4 = ttk.LabelFrame(self, text="4. DBSCAN (Bereinigung)")
        frame4.pack(fill="x", padx=10, pady=5)
        ttk.Button(frame4, text="ROI zuschneiden & DBSCAN ausführen", command=self.run_dbscan).pack(pady=10)

        # --- 5. Tracking ---
        frame5 = ttk.LabelFrame(self, text="5. ICP Tracking (Point-to-Plane)")
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

        # --- 6. Tracking Analyse ---
        frame6 = ttk.LabelFrame(self, text="🚀 Analyse Tracking CSV")
        frame6.pack(fill="x", padx=10, pady=5)
        ttk.Button(frame6, text="CSV Ergebnisse Plotten", command=self.plot_tracking_results).pack(pady=10)

        # --- Progress Bar ---
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=10)
        self.lbl_status = ttk.Label(self, text="Status: Bereit")
        self.lbl_status.pack(pady=5)

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("HDF5 Files", "*.h5")])
        if not path: return
        self.h5_path = path
        self.lbl_file.config(text=os.path.basename(path))
        self.lbl_status.config(text="Status: Lade H5 Daten...")
        self.update()

        def task():
            try:
                if self.stream is not None:
                    self.stream.close()

                self.stream = H5PointCloudStream(Path(self.h5_path))
                self.frames = []
                self.timestamps = []
                self.pcds = []
                self.progress["maximum"] = self.stream.num_frames

                for i in range(self.stream.num_frames):
                    pcd = self.stream.get_pcd(i, voxel_size=1.5)
                    self.pcds.append(pcd)
                    self.frames.append(np.asarray(pcd.points))
                    ts_steady, _ = self.stream.get_timestamps(i)
                    self.timestamps.append(str(ts_steady))

                    if i % 10 == 0:
                        self.progress["value"] = i + 1
                        self.update_idletasks()

                # Thread-safe UI Updates
                self.after(0, lambda: self.slider.config(to=len(self.pcds) - 1))
                self.after(0, lambda: self.lbl_status.config(text=f"Status: {len(self.pcds)} Frames geladen."))
                self.after(0, lambda: self.on_slider_change(0))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Fehler", f"Konnte H5 nicht laden:\n{e}"))

        threading.Thread(target=task, daemon=True).start()

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

            # Punktwolke hinzufügen
            self.pcd_vis = o3d.geometry.PointCloud()
            if len(self.pcds[self.current_frame].points) > 0:
                self.pcd_vis.points = self.pcds[self.current_frame].points
                if self.pcds[self.current_frame].has_colors():
                    self.pcd_vis.colors = self.pcds[self.current_frame].colors
            else:
                self.pcd_vis.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))
            self.vis.add_geometry(self.pcd_vis)

            # NEU: Leere Bounding Box (LineSet) hinzufügen
            self.bbox_vis = o3d.geometry.LineSet()
            self.vis.add_geometry(self.bbox_vis)

            # Kamera auf Zentrum setzen
            self.vis.reset_view_point(True)

        self.update_viewer_geometry()

    def update_viewer_geometry(self):
        if self.vis and self.frames:
            pcd_current = self.pcds[self.current_frame]

            if len(pcd_current.points) == 0:
                self.lbl_status.config(text="Status: Warnung - Aktueller Frame hat 0 Punkte!")
                return

            self.pcd_vis.points = pcd_current.points
            if pcd_current.has_colors():
                self.pcd_vis.colors = pcd_current.colors

            self.vis.update_geometry(self.pcd_vis)
            self.vis.poll_events()
            self.vis.update_renderer()

    def poll_o3d(self):
        if self.vis is not None:
            self.vis.poll_events()
            self.vis.update_renderer()
        self.after(50, self.poll_o3d)

    # ---------------------------------------------------------
    # 3. ROI (Neu: Rendert nur eine schwarze Box)
    # ---------------------------------------------------------
    def apply_roi_current(self):
        if not self.frames: return
        x_min, x_max = self.roi_vars['x_min'].get(), self.roi_vars['x_max'].get()
        y_min, y_max = self.roi_vars['y_min'].get(), self.roi_vars['y_max'].get()
        z_min, z_max = self.roi_vars['z_min'].get(), self.roi_vars['z_max'].get()

        # Mathematische Bounding Box erstellen und speichern
        self.roi_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(x_min, y_min, z_min),
            max_bound=(x_max, y_max, z_max)
        )

        # Visuelles Drahtgitter-Modell (LineSet) erstellen
        if self.bbox_vis is not None:
            lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(self.roi_bbox)
            lines.paint_uniform_color([0, 0, 0])  # Schwarz rendern

            # Update des existierenden LineSets im Viewer
            self.bbox_vis.points = lines.points
            self.bbox_vis.lines = lines.lines
            self.bbox_vis.colors = lines.colors
            self.vis.update_geometry(self.bbox_vis)

        self.lbl_status.config(text="Status: ROI-Box Vorschau aktiv. (Punkte sind noch da)")

    def save_roi(self):
        if not self.h5_path:
            messagebox.showwarning("Warnung", "Bitte lade zuerst eine H5-Datei!")
            return

        if self.roi_bbox is None:
            messagebox.showwarning("Warnung",
                                   "Bitte klicke zuerst auf 'ROI als Box einblenden', um die Box zu definieren!")
            return

        # Dateiname aus der H5-Datei generieren
        current_time = time.strftime("%H%M%S")
        h5_basename = os.path.splitext(os.path.basename(self.h5_path))[0]
        roi_filename = f"roi_{h5_basename}_{current_time}.json"

        # Koordinaten auslesen
        min_b = self.roi_bbox.get_min_bound()
        max_b = self.roi_bbox.get_max_bound()

        roi_data = {
            "x": [float(min_b[0]), float(max_b[0])],
            "y": [float(min_b[1]), float(max_b[1])],
            "z": [float(min_b[2]), float(max_b[2])]
        }

        try:
            with open(roi_filename, 'w') as f:
                json.dump(roi_data, f, indent=4)
            self.lbl_status.config(text=f"Status: ROI erfolgreich gespeichert als {roi_filename}")
            messagebox.showinfo("Erfolg", f"ROI wurde gespeichert als:\n{roi_filename}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte ROI nicht speichern:\n{e}")
    # ---------------------------------------------------------
    # 4. DBSCAN (Neu: Wendet ROI *hier* erst hart an)
    # ---------------------------------------------------------
    def run_dbscan(self):
        if not self.frames: return

        def task():
            self.lbl_status.config(text="Status: Beschneide mit ROI und führe DBSCAN aus...")
            self.progress["maximum"] = len(self.pcds)

            for i, pcd in enumerate(self.pcds):
                # 1. ZUERST auf ROI zuschneiden, falls definiert
                if self.roi_bbox is not None:
                    pcd = pcd.crop(self.roi_bbox)

                # 2. DANN DBSCAN ausführen
                labels = np.array(pcd.cluster_dbscan(eps=20.0, min_points=10, print_progress=False))
                if len(labels) == 0:
                    self.pcds[i] = pcd
                    continue

                max_label = labels.max()
                if max_label >= 0:
                    counts = np.bincount(labels[labels >= 0])
                    phantom_label = np.argmax(counts)
                    phantom_indices = np.where(labels == phantom_label)[0]
                    self.pcds[i] = pcd.select_by_index(phantom_indices)

                if i % 10 == 0:
                    self.progress["value"] = i + 1
                    self.update_idletasks()

            self.lbl_status.config(text="Status: ROI & DBSCAN abgeschlossen. (Viewer updatet jetzt)")
            # Viewer updaten, damit man das saubere Ergebnis sieht
            self.after(0, self.update_viewer_geometry)

        threading.Thread(target=task, daemon=True).start()

    # ---------------------------------------------------------
    # 5. Tracking (Sichert ab, falls DBSCAN übersprungen wurde)
    # ---------------------------------------------------------
    def start_tracking(self):
        if not self.frames: return

        start_str = self.var_start_frame.get().strip()
        end_str = self.var_end_frame.get().strip()

        start_idx = int(start_str) if start_str else 0
        end_idx = int(end_str) if end_str else len(self.pcds) - 1

        # --- ERSETZEN IN start_tracking ---
        current_time = time.strftime("%H%M%S")
        h5_basename = os.path.splitext(os.path.basename(self.h5_path))[0]
        default_name = f"ICP_{h5_basename}_{current_time}.csv"

        save_path = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not save_path: return

        def task():
            self.lbl_status.config(text=f"Status: Tracking (Point-to-Plane) von Frame {start_idx} bis {end_idx}...")
            self.progress["maximum"] = end_idx - start_idx

            results = []

            # Referenz Frame
            ref_pcd = self.pcds[start_idx]
            # Zur Sicherheit ROI anwenden, falls DBSCAN nicht geklickt wurde
            if self.roi_bbox is not None:
                ref_pcd = ref_pcd.crop(self.roi_bbox)

            ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

            last_phantom_trans = np.eye(4)

            for i in range(start_idx, end_idx + 1):
                source_pcd = self.pcds[i]
                ts = self.timestamps[i]

                # Auch hier: Zur Sicherheit ROI anwenden
                if self.roi_bbox is not None:
                    source_pcd = source_pcd.crop(self.roi_bbox)

                source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
                init_guess = np.linalg.inv(last_phantom_trans)

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    source_pcd, ref_pcd, 30.0, init_guess,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )

                phantom_trans = np.linalg.inv(reg_p2p.transformation)
                last_phantom_trans = phantom_trans

                tx, ty, tz = phantom_trans[:3, 3]
                euler = R.from_matrix(phantom_trans[:3, :3]).as_euler('xyz')

                results.append({
                    "Timestamp (steady)": ts,
                    "Current Tx": tx, "Ty": ty, "Tz": tz,
                    "Rx": euler[0], "Ry": euler[1], "Rz": euler[2],
                    "Score": reg_p2p.fitness * 100, "RMSE3D": reg_p2p.inlier_rmse
                })

                if i % 5 == 0:
                    self.progress["value"] = i - start_idx + 1
                    self.update_idletasks()

            df = pd.DataFrame(results)

            # Neue Logik mit Header und ROI
            with open(save_path, 'w', encoding='utf-8') as f:
                # Zeile 1: Pfad
                f.write(f"H5 Tracking Path:,{self.h5_path},,,,,,\n")
                # Zeile 2: Header für ROI
                f.write("ROI Config,Axis,Min,Max,,,,,\n")

                # Zeile 3-5: ROI Koordinaten (falls vorhanden)
                if self.roi_bbox is not None:
                    min_b = self.roi_bbox.get_min_bound()
                    max_b = self.roi_bbox.get_max_bound()
                    f.write(f",X,{min_b[0]:.2f},{max_b[0]:.2f},,,,,\n")
                    f.write(f",Y,{min_b[1]:.2f},{max_b[1]:.2f},,,,,\n")
                    f.write(f",Z,{min_b[2]:.2f},{max_b[2]:.2f},,,,,\n")
                else:
                    f.write(",X,None,None,,,,,\n")
                    f.write(",Y,None,None,,,,,\n")
                    f.write(",Z,None,None,,,,,\n")

                # Danach das eigentliche Pandas DataFrame schreiben
                df.to_csv(f, index=False, lineterminator='\n')
            self.lbl_status.config(text=f"Status: Tracking beendet. Gespeichert in {os.path.basename(save_path)}")

        threading.Thread(target=task, daemon=True).start()

    # ---------------------------------------------------------
    # 6. Tracking Analyse (CSV Plotter)
    # ---------------------------------------------------------
    def plot_tracking_results(self):
        csv_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not csv_path: return
        # Überspringe dynamisch die Meta-Zeilen, bis der Header "Timestamp" gefunden wird
        skip_lines = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if "Timestamp (steady)" in line or "Current Tx" in line:
                    skip_lines = i
                    break

        df = pd.read_csv(csv_path, skiprows=skip_lines)
        plot_win = tk.Toplevel(self)
        plot_win.title(f"Tracking Analyse - {os.path.basename(csv_path)}")
        plot_win.geometry("900x700")

        controls = ttk.Frame(plot_win)
        controls.pack(fill="x", padx=10, pady=5)

        show_trans = tk.BooleanVar(value=True)
        show_rot = tk.BooleanVar(value=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        canvas = FigureCanvasTkAgg(fig, master=plot_win)

        def update_plot():
            ax1.clear()
            ax2.clear()
            x_vals = range(len(df))

            if show_trans.get():
                if "Current Tx" in df.columns:
                    ax1.plot(x_vals, df["Current Tx"], label="Tx", linestyle='-')
                    ax1.plot(x_vals, df["Ty"], label="Ty", linestyle='-')
                    ax1.plot(x_vals, df["Tz"], label="Tz", linestyle='-')
            if show_rot.get():
                if "Rx" in df.columns:
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