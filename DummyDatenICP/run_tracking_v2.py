import numpy as np
import pandas as pd
import open3d as o3d
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from extractH5.h5_loader import H5PointCloudStream
from datetime import datetime

# --- NEU: Den interaktiven ROI-Finder importieren ---
from find_roi import run_interactive_roi

# --- SETUP ---
h5_path = Path("/Volumes/INTENSO/1775668627720/TrackingLog.h5")
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)


def get_phantom_movement(source_pcd, ref_pcd, init_trans):
    """
    Berechnet die Bewegung des Phantoms mittels Point-to-Plane ICP.
    """
    source_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))
    ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source_pcd, ref_pcd, 30.0, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    return np.linalg.inv(result.transformation), result.fitness, result.inlier_rmse


def main():
    # 1. ROI Interaktiv abfragen! (Pausiert hier, bis du fertig bist)
    print("Starte ROI-Konfiguration...")
    roi_bbox = run_interactive_roi(h5_path)

    if roi_bbox is None:
        print("Tracking wurde vorzeitig beendet, da keine ROI definiert wurde.")
        sys.exit(0)

    # 2. Start des eigentlichen Trackings
    print("\n" + "=" * 40)
    print("🚀 Starte ICP Tracking...")
    results = []
    with H5PointCloudStream(h5_path) as stream:
        # Referenz laden (nutzt jetzt die BoundingBox, die du gerade definiert hast)
        ref_pcd = stream.get_pcd(0, roi_bbox=roi_bbox, voxel_size=1.5)

        last_phantom_trans = np.eye(4)

        for i in range(stream.num_frames):
            source_pcd = stream.get_pcd(i, roi_bbox=roi_bbox, voxel_size=1.5)
            _, ts_steady = stream.get_timestamps(i)

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
            if i % 50 == 0:
                print(f"Frame {i}/{stream.num_frames} verarbeitet...")

    # Output CSV dynamisch nach der H5-Datei und mit Zeitstempel benennen
    timestamp = datetime.now().strftime("%H%M%S")
    csv_filename = f"tracking_v2_{h5_path.stem}_{timestamp}.csv"
    pd.DataFrame(results).to_csv(output_dir / csv_filename, index=False)
    print(f"✅ Tracking v2 beendet. Datei '{csv_filename}' erstellt in {output_dir}/")


if __name__ == "__main__":
    main()