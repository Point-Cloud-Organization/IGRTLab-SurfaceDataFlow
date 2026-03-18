import numpy as np
import open3d as o3d
import json
from pathlib import Path
from extractH5.h5_loader import H5PointCloudStream

# --- PFADE ANPASSEN ---
h5_path = Path("/Users/timjb/PycharmProjects/Point_Cloud/DemoData/record.h5")
roi_path = Path("Calibration/roi_config.json")

# ROI laden
with open(roi_path, "r") as f:
    roi_data = json.load(f)

roi_bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=roi_data["min_bound"],
    max_bound=roi_data["max_bound"]
)


def main():
    with H5PointCloudStream(h5_path) as stream:
        print("Lade Frame 0 in voller Auflösung...")
        # WICHTIG: voxel_size=None, damit wir die echten rohen Daten haben!
        pcd = stream.get_pcd(0, roi_bbox=roi_bbox, voxel_size=None)
        points = np.asarray(pcd.points)

        # 1. Den geometrischen Mittelpunkt des Phantoms finden
        center = pcd.get_center()

        # 2. Eine hauchdünne "Scheibe" aus der Mitte herausschneiden
        # Wir betrachten nur Punkte, die maximal 1 mm nach oben/unten (Y)
        # und vorne/hinten (Z) abweichen.
        y_mask = (points[:, 1] > center[1] - 1.0) & (points[:, 1] < center[1] + 1.0)
        z_mask = (points[:, 2] > center[2] - 1.0) & (points[:, 2] < center[2] + 1.0)

        slice_points = points[y_mask & z_mask]

        if len(slice_points) < 2:
            print("❌ Nicht genug Punkte in der Scheibe gefunden. Bitte ROI prüfen.")
            return

        # 3. Die X-Koordinaten dieser Scheibe isolieren und sortieren
        x_coords = np.sort(slice_points[:, 0])

        # 4. Die Abstände zwischen benachbarten X-Werten berechnen
        x_diffs = np.diff(x_coords)

        # 5. Wir filtern winzige "Aufrauhungen" (< 0.1 mm) und Löcher (> 10 mm) heraus
        valid_diffs = x_diffs[(x_diffs > 0.1) & (x_diffs < 10.0)]

        if len(valid_diffs) == 0:
            print("❌ Keine auswertbaren Abstände gefunden.")
            return

        print("-" * 30)
        print(f"Ausgewertete Punkte auf der X-Linie: {len(valid_diffs)}")
        print(f"Mittlerer X-Abstand: {np.mean(valid_diffs):.4f} mm")
        print(f"Median X-Abstand:  {np.median(valid_diffs):.4f} mm")
        print(f"Minimaler X-Abstand: {np.min(valid_diffs):.4f} mm")
        print(f"Maximaler X-Abstand: {np.max(valid_diffs):.4f} mm")


if __name__ == "__main__":
    main()