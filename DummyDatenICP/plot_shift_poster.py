import open3d as o3d
import numpy as np
import json
from pathlib import Path
from extractH5.h5_loader import H5PointCloudStream

# --- SETUP ---
h5_path = Path("/Users/timjb/Documents/MUI/Radioonko/Daten/DemoData/record.h5")
# WICHTIG: Trag hier den exakten Namen deiner erstellten ROI-JSON ein!
roi_path = Path("Calibration/roi_record_3003.json")

# Welcher Frame soll mit Frame 0 (Referenz) verglichen werden?
# (Such dir den Frame aus, wo die 10mm Verschiebung am deutlichsten ist, z.B. 150)
SHIFT_FRAME_INDEX = 300


def extract_main_cluster(pcd, eps=15.0, min_points=50):
    """
    Entfernt "fliegende" Punkte durch DBSCAN Clustering.
    Behält nur den größten zusammenhängenden Punktewolken-Cluster.
    eps: Maximaler Abstand in mm, damit Punkte als zusammenhängend gelten.
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

    if len(labels) == 0:
        return pcd

    max_label = labels.max()
    if max_label < 0:  # Alle Punkte sind Rauschen
        return pcd

    # Zähle, welches Label (welcher Cluster) die meisten Punkte hat
    counts = np.bincount(labels[labels >= 0])
    largest_cluster_idx = counts.argmax()

    # Extrahiere nur die Punkte des größten Clusters
    main_cluster_indices = np.where(labels == largest_cluster_idx)[0]
    pcd_clean = pcd.select_by_index(main_cluster_indices)

    return pcd_clean


def main():
    # 1. ROI Laden (Dient jetzt nur noch der Visualisierung!)
    with open(roi_path, "r") as f:
        roi_data = json.load(f)
    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(roi_data["min_bound"], roi_data["max_bound"])
    roi_bbox.color = (0.2, 0.8, 0.2)  # Schönes Grün für die Box

    with H5PointCloudStream(h5_path) as stream:
        # 2. Referenz Frame (0) laden -> WICHTIG: roi_bbox weggelassen!
        print("Lade komplette Referenz (Frame 0)...")
        pcd_ref = stream.get_pcd(0, voxel_size=1.5)  # <--- HIER GEÄNDERT
        pcd_ref = extract_main_cluster(pcd_ref)
        pcd_ref.paint_uniform_color([0.1, 0.5, 0.8])  # Poster-Blau
        pcd_ref.estimate_normals()

        # 3. Verschobenen Frame laden -> WICHTIG: roi_bbox weggelassen!
        print(f"Lade komplettes verschobenes Phantom (Frame {SHIFT_FRAME_INDEX})...")
        pcd_shift = stream.get_pcd(SHIFT_FRAME_INDEX, voxel_size=1.5)  # <--- HIER GEÄNDERT
        pcd_shift = extract_main_cluster(pcd_shift)
        pcd_shift.paint_uniform_color([0.9, 0.4, 0.1])  # Poster-Orange
        pcd_shift.estimate_normals()

        # 4. Visualisierung (Hier geben wir die Box wieder dazu)
    print("\n--- Viewer für Poster-Screenshot geöffnet ---")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Phantom Shift Visualization", width=1920, height=1080)

    vis.add_geometry(pcd_ref)
    vis.add_geometry(pcd_shift)
    vis.add_geometry(roi_bbox)  # <--- Box wird on top gerendert


    # Hintergrund auf Weiß stellen (besser für Poster)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 3.0  # Punkte etwas größer machen

    # Beide Punktwolken zu einer einzigen fusionieren
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined += pcd_ref
    pcd_combined += pcd_shift

    # Als PLY für Sketchfab exportieren
    export_path = "output/poster_3d_modell.ply"
    o3d.io.write_point_cloud(export_path, pcd_combined)
    print(f"✅ 3D-Modell für Sketchfab exportiert: {export_path}")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()