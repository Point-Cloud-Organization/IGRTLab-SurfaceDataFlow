import numpy as np
import pandas as pd
import open3d as o3d
import json
import os
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from extractH5.h5_loader import H5PointCloudStream

# --- SETUP ---
# Pfade anpassen (Nutze den Pfad, der in find_roi.py geklappt hat)
h5_path = Path("/Volumes/INTENSO/01_Data/01_ETD/hd5/patients/1768817211649/TrackingLog.h5")
roi_path = Path("Calibration/roi_config_1768817211649.json")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 1. ROI Laden
if not roi_path.exists():
    raise FileNotFoundError("Bitte erst find_roi.py ausführen und ROI speichern!")

with open(roi_path, "r") as f:
    roi_data = json.load(f)

roi_bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=roi_data["min_bound"],
    max_bound=roi_data["max_bound"]
)

# 2. ICP Parameter
# voxel_size=2.0 sorgt für Speed, ohne die Präzision zu verlieren
VOXEL_SIZE = 2.0
DISTANCE_THRESHOLD = 15.0


def get_euler_from_matrix(matrix):
    """Extrahiert XYZ-Euler-Winkel (Radiant) aus der 4x4 Matrix."""
    rotation_matrix = matrix[:3, :3]
    return R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)


def main():
    results = []

    with H5PointCloudStream(h5_path) as stream:
        # Referenz-Punktwolke (Frame 0) vorbereiten
        print(f"Lade Referenz-Frame (0)...")
        ref_pcd = stream.get_pcd(0, roi_bbox=roi_bbox, voxel_size=VOXEL_SIZE)

        # WICHTIG: Die Schätzung für Frame 0 ist die Identitätsmatrix
        current_trans = np.eye(4)

        print(f"Starte ICP-Tracking für {stream.num_frames} Frames...")

        for i in range(stream.num_frames):
            # Aktuellen Frame laden
            source_pcd = stream.get_pcd(i, roi_bbox=roi_bbox, voxel_size=VOXEL_SIZE)
            _, ts_steady = stream.get_timestamps(i)

            # ICP Algorithmus (Point-to-Point)
            # Nutzt die Transformation des letzten Frames als Startwert (Warm Start)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_pcd, ref_pcd, DISTANCE_THRESHOLD, current_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )

            # Transformation speichern für den nächsten Frame
            current_trans = reg_p2p.transformation

            # Werte extrahieren
            tx, ty, tz = current_trans[:3, 3]
            rx, ry, rz = get_euler_from_matrix(current_trans)

            # Ergebnis-Zeile erstellen (Format wie in deiner CSV)
            results.append({
                "Timestamp (steady)": ts_steady,
                "Current Tx": tx, "Ty": ty, "Tz": tz,
                "Rx": rx, "Ry": ry, "Rz": rz,
                "Score": reg_p2p.fitness * 100,
                "RMSE3D": reg_p2p.inlier_rmse
            })

            if i % 50 == 0:
                print(f"Fortschritt: {i}/{stream.num_frames} Frames verarbeitet.")

    # 3. Speichern
    df = pd.DataFrame(results)
    csv_filename = output_dir / "failedFirstTry2.csv"
    df.to_csv(csv_filename, index=False)

    print("-" * 30)
    print(f"✅ Fertig! CSV wurde gespeichert unter: {csv_filename}")
    print(f"Durchschnittlicher RMSE: {df['RMSE3D'].mean():.4f} mm")


if __name__ == "__main__":
    main()