import numpy as np
import pandas as pd
import open3d as o3d
import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from extractH5.h5_loader import H5PointCloudStream

# --- SETUP ---
#h5_path = Path("/Users/timjb/PycharmProjects/Point_Cloud/DemoData/record.h5")
h5_path = '/Volumes/INTENSO/01_Data/01_ETD/hd5/patients/1768810376373/TrackingLog.h5'
roi_path = Path("Calibration/roi_config_1768810376373.json")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

with open(roi_path, "r") as f:
    roi_data = json.load(f)
roi_bbox = o3d.geometry.AxisAlignedBoundingBox(roi_data["min_bound"], roi_data["max_bound"])


def get_phantom_movement(source_pcd, ref_pcd, init_trans):
    """
    Berechnet die Bewegung des Phantoms mittels Point-to-Plane ICP.
    """
    # 1. Normalen schätzen (zwingend erforderlich für Point-to-Plane)
    # Wir schauen im Umkreis von 5mm nach der Oberflächenneigung
    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

    # 2. Point-to-Plane ICP (Viel robuster gegen "Abhauen")
    # Wir nutzen einen sehr großzügigen Suchradius von 30mm für den Fang
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, ref_pcd, 30.0, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # 3. WICHTIG: Die Matrix invertieren!
    # ICP gibt uns: Wie komme ich von Source zu Ref? (Korrektur)
    # Wir wollen: Wie hat sich das Phantom von Ref zu Source bewegt?
    return np.linalg.inv(result.transformation), result.fitness, result.inlier_rmse


def main():
    results = []
    with H5PointCloudStream(h5_path) as stream:
        # Referenz laden
        ref_pcd = stream.get_pcd(0, roi_bbox=roi_bbox, voxel_size=1.5)

        # Startwert: Keine Bewegung
        last_phantom_trans = np.eye(4)

        for i in range(stream.num_frames):
            source_pcd = stream.get_pcd(i, roi_bbox=roi_bbox, voxel_size=1.5)
            _, ts_steady = stream.get_timestamps(i)

            # ICP berechnen (Wir nutzen das Ergebnis vom letzten Frame als Startwert)
            # Aber wir müssen es für den ICP wieder invertieren (als Korrektur-Vorschlag)
            init_guess = np.linalg.inv(last_phantom_trans)

            phantom_trans, score, rmse = get_phantom_movement(source_pcd, ref_pcd, init_guess)
            last_phantom_trans = phantom_trans

            tx, ty, tz = phantom_trans[:3, 3]
            euler = R.from_matrix(phantom_trans[:3, :3]).as_euler('xyz')

            results.append({
                "Timestamp (steady)": ts_steady,
                "Current Tx": tx, "Ty": ty, "Tz": tz,
                "Rx": euler[0], "Ry": euler[1], "Rz": euler[2],
                "Score": score * 100, "RMSE3D": rmse
            })
            if i % 50 == 0: print(f"Frame {i} processed...")

    pd.DataFrame(results).to_csv(output_dir / "tracking_v2.csv", index=False)
    print("✅ Tracking v2 beendet. Datei 'tracking_v2.csv' erstellt.")


if __name__ == "__main__":
    main()