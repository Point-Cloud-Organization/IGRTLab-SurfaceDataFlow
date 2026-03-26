import open3d as o3d
import numpy as np
import imageio
import json
from pathlib import Path
from extractH5.h5_loader import H5PointCloudStream

# --- SETUP ---
h5_path = Path(r"C:\Users\timjb\iCloudDrive\Documents\MUI\Radioonko\Daten\DemoData\record.h5")
roi_path = Path("Calibration/roi_record.json")  # Dein ROI Name
output_gif = "phantom_tracking.gif"

# Wie viele Frames überspringen? (z.B. step=5 macht das GIF schneller und kleiner)
FRAME_STEP = 5
MAX_FRAMES = 200  # Beschränkung, damit das GIF nicht gigantisch wird


def extract_main_cluster(pcd, eps=15.0, min_points=50):
    """Dein bewährter Rausch-Filter aus dem vorherigen Skript"""
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if len(labels) == 0: return pcd
    max_label = labels.max()
    if max_label < 0: return pcd
    counts = np.bincount(labels[labels >= 0])
    pcd_clean = pcd.select_by_index(np.where(labels == counts.argmax())[0])
    return pcd_clean


def main():
    with open(roi_path) as f:
        roi_bbox = o3d.geometry.AxisAlignedBoundingBox(**{k: json.load(f)[k] for k in ["min_bound", "max_bound"]})

    print("🎥 Bereite Video-Rendering vor...")
    frames_images = []

    # 1. Unsichtbares/Automatisiertes Fenster erstellen
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, visible=True)

    # Render-Optionen (Weißer Hintergrund für Poster-Style)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])
    opt.point_size = 4.0

    with H5PointCloudStream(h5_path) as stream:
        # Start-Geometrie laden, damit die Kamera weiß, wo sie hinschauen muss
        pcd = stream.get_pcd(0, roi_bbox=roi_bbox, voxel_size=1.5)
        pcd = extract_main_cluster(pcd)
        pcd.paint_uniform_color([0.1, 0.5, 0.8])  # Blau

        vis.add_geometry(pcd)

        # Kamera einmalig zentrieren
        vis.poll_events()
        vis.update_renderer()

        print("📸 Starte Aufnahme der Frames (Kamera festhalten!)...")
        limit = min(stream.num_frames, MAX_FRAMES)

        for i in range(0, limit, FRAME_STEP):
            # Neuen Frame laden & bereinigen
            new_pcd = stream.get_pcd(i, roi_bbox=roi_bbox, voxel_size=1.5)
            new_pcd = extract_main_cluster(new_pcd)
            new_pcd.paint_uniform_color([0.1, 0.5, 0.8])

            # Punkte der vorhandenen Geometrie überschreiben (hält die Kamera fest!)
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors

            # Bild aktualisieren
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Screenshot in den RAM speichern
            img = vis.capture_screen_float_buffer(do_render=True)
            img_array = (np.asarray(img) * 255).astype(np.uint8)
            frames_images.append(img_array)

            print(f"Frame {i}/{limit} gerendert...", end="\r")

    vis.destroy_window()

    # 2. Aus den Screenshots ein GIF bauen
    print(f"\n💾 Speichere GIF als {output_gif} (Das kann kurz dauern)...")
    # fps=15 sorgt für eine flüssige, aber nicht zu schnelle Wiedergabe
    imageio.mimsave(output_gif, frames_images, fps=15)
    print("✅ GIF erfolgreich erstellt!")


if __name__ == "__main__":
    main()
