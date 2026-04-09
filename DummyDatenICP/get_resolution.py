import numpy as np
import open3d as o3d
import json
from pathlib import Path
from extractH5.h5_loader import H5PointCloudStream

# Pfade anpassen
h5_path = Path("/archive/DemoData/record.h5")
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
        # 1. Frame laden: WICHTIG -> voxel_size=None!
        # Wir wollen die rohen, un-downsampled Punkte sehen.
        print("Lade Frame 0 in voller Auflösung...")
        pcd = stream.get_pcd(0, roi_bbox=roi_bbox, voxel_size=None)

        anzahl_punkte = len(pcd.points)
        print(f"Anzahl Punkte auf dem Phantom: {anzahl_punkte}")

        # 2. Abstand zum nächsten Nachbarn für JEDEN Punkt berechnen
        print("Berechne Abstände...")
        distances = pcd.compute_nearest_neighbor_distance()
        distances_np = np.asarray(distances)

        # 3. Statistik ausgeben
        avg_dist = np.mean(distances_np)
        min_dist = np.min(distances_np)
        max_dist = np.max(distances_np)

        print("-" * 30)
        print(f"Durchschnittlicher Punktabstand: {avg_dist:.4f} mm")
        print(f"Minimaler Abstand (dichteste Stelle): {min_dist:.4f} mm")
        print(f"Maximaler Abstand (Löcher/Ränder): {max_dist:.4f} mm")


if __name__ == "__main__":
    main()