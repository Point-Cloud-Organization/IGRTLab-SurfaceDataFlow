import open3d as o3d
import json
from pathlib import Path
from extractH5.h5_loader import H5PointCloudStream

# --- KONFIGURATION ---
h5_file = Path("/Users/timjb/PycharmProjects/Point_Cloud/record.h5")  # Pfad anpassen
config_file = Path("roi_config.json")

# Deine aktuellen Test-Werte (in mm)
min_x, min_y, min_z = [-100, -300, -50]
max_x, max_y, max_z = [100,  0,  300]


def main():
    with H5PointCloudStream(h5_file) as stream:
        pcd = stream.get_pcd(0)

    # ROI Box für Visualisierung erstellen
    roi = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds, max_bound=max_bounds)
    roi.color = (1, 0, 0)  # Rote Box

    # Punkte in der Box rot färben
    pcd_cropped = pcd.crop(roi)
    pcd_cropped.paint_uniform_color([1, 0, 0])
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    print("--- Viewer geöffnet ---")
    print("Prüfe die rote Box. Schließe das Fenster, um fortzufahren.")
    o3d.visualization.draw_geometries([pcd, pcd_cropped, roi])

    # Abfrage im Terminal
    answer = input("\nSoll diese ROI gespeichert werden? (j/n): ").lower()
    if answer == 'j':
        config = {
            "min_bound": min_bounds,
            "max_bound": max_bounds,
            "description": "ROI für Phantom-Tracking"
        }
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
        print(f"✅ ROI erfolgreich in {config_file} gespeichert!")
    else:
        print("❌ Speichern abgebrochen.")


if __name__ == "__main__":
    main()