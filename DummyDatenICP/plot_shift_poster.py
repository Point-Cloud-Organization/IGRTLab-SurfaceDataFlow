import numpy as np
import open3d as o3d
import json
import sys
from pathlib import Path

# --- Import deines eigenen Loaders ---
try:
    from extractH5.h5_loader import H5PointCloudStream
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent / "DummyDatenICP"))
    from extractH5.h5_loader import H5PointCloudStream


def extract_main_phantom(pcd, eps=20.0, min_points=10):
    """Führt DBSCAN aus und gibt nur den größten Cluster zurück."""
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if len(labels) == 0:
        return pcd
    max_label = labels.max()
    if max_label >= 0:
        counts = np.bincount(labels[labels >= 0])
        phantom_label = np.argmax(counts)
        phantom_indices = np.where(labels == phantom_label)[0]
        return pcd.select_by_index(phantom_indices)
    return pcd


def main():
    # --- KONFIGURATION ---
    h5_path = Path("/Users/timjb/Documents/MUI/Radioonko/Daten/1775668627720/TrackingLog.h5")
    roi_path = Path("roi_TrackingLog.json")  # Deine in der GUI gespeicherte ROI

    frame_x1 = 40  # Basis-Frame (gesamter roher Raum)
    frame_x2 = 110  # Verschobener Frame (nur ROI Ausschnitt + DBSCAN)

    # Farben (RGB)
    color_base = [0.1, 0.4, 0.8]  # Blau für den kompletten Raum
    color_shift = [1.0, 0.5, 0.0]  # Orange für die verschobene ROI

    output_filename = f"poster_model_{h5_path.stem}.ply"

    print(f"Lade ROI aus {roi_path}...")
    with open(roi_path, 'r') as f:
        roi_data = json.load(f)

    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(roi_data["x"][0], roi_data["y"][0], roi_data["z"][0]),
        max_bound=(roi_data["x"][1], roi_data["y"][1], roi_data["z"][1])
    )

    print("Öffne H5 Stream...")
    with H5PointCloudStream(h5_path) as stream:
        # ----------------------------------------------------
        # SCHRITT 1: Basis-Frame (x1) - Das KOMPLETTE Roh-Bild
        # ----------------------------------------------------
        print(f"Verarbeite Frame {frame_x1} (Basis - Komplett Roh)...")
        # Wir laden Frame 1 komplett ohne Crop.
        pcd_base = stream.get_pcd(frame_x1, voxel_size=1.5)

        # WICHTIG: Kein DBSCAN mehr für Frame 1!
        # Direkt blau einfärben (überschreibt die z-Achsen-Farben aus der Rohdatei)
        pcd_base.paint_uniform_color(color_base)

        # ----------------------------------------------------
        # SCHRITT 2: Verschobener Frame (x2) - Nur der ROI Bereich
        # ----------------------------------------------------
        print(f"Verarbeite Frame {frame_x2} (Verschiebung - ROI & DBSCAN)...")
        # Hier wie gehabt: Gleich beim Laden auf die Box zuschneiden
        pcd_shift_raw = stream.get_pcd(frame_x2, roi_bbox=roi_bbox, voxel_size=1.5)

        # Hier ist DBSCAN gewollt, damit die orange Box saubere Ränder ohne Rauschen hat
        pcd_shift_clean = extract_main_phantom(pcd_shift_raw)

        # Orange einfärben
        pcd_shift_clean.paint_uniform_color(color_shift)

    # ----------------------------------------------------
    # SCHRITT 3: Zusammenführen und Speichern
    # ----------------------------------------------------
    print("Füge Punktwolken zusammen...")
    poster_pcd = pcd_base + pcd_shift_clean

    print(f"Speichere {output_filename} für Sketchfab...")
    o3d.io.write_point_cloud(output_filename, poster_pcd, write_ascii=False)

    print("✅ Fertig! Modell gespeichert.")

    # Direkte Vorschau öffnen
    o3d.visualization.draw_geometries([poster_pcd], window_name="Poster PLY Vorschau")


if __name__ == "__main__":
    main()