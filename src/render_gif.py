import open3d as o3d
import numpy as np
import imageio
import json
import sys
from pathlib import Path

# --- Import deines eigenen Loaders ---
try:
    from extractH5.h5_loader import H5PointCloudStream
except ImportError:
    # Navigiere aus 'src' heraus (parent.parent) und dann in 'data/DummyDatenICP'
    project_root = Path(__file__).resolve().parent.parent
    dummy_data_path = project_root / "data" / "DummyDatenICP"

    sys.path.append(str(dummy_data_path))
    from extractH5.h5_loader import H5PointCloudStream

# --- SETUP ---
h5_path = Path("/Users/timjb/Documents/MUI/Radioonko/Daten/1775668627720/TrackingLog.h5")
roi_path = Path("/Users/timjb/PycharmProjects/Point_Cloud/data/DummyDatenICP/roi_TrackingLog_124758.json")
output_video = "surf_analysis_highres.mp4"
fps = 15

# Frame-Bereich für das Video
START_FRAME = 40
END_FRAME = 180


def main():
    with open(roi_path, "r") as f:
        roi_data = json.load(f)

    # ROI als scharfen roten Rahmen vorbereiten
    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(roi_data["min_bound"], roi_data["max_bound"])
    roi_box_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(roi_bbox)
    roi_box_lines.paint_uniform_color([1, 0, 0])
    roi_center = roi_bbox.get_center()

    with H5PointCloudStream(h5_path) as stream:
        # Initialen Frame für Kamera-Setup laden (Ohne Clustering!)
        # Voxel_size auf 1.5 oder 2.0 setzen, damit es schneller lädt und rendert
        pcd = stream.get_pcd(START_FRAME, voxel_size=1.5)
        pcd.paint_uniform_color([0.1, 0.5, 0.8])
        pcd.estimate_normals()

        # ==========================================
        # PHASE 1: INTERAKTIVES KAMERA-SETUP
        # ==========================================
        print("\n" + "=" * 50)
        print("📸 KAMERA SETUP")
        print("1. Ein Fenster öffnet sich jetzt.")
        print("2. Positioniere die Punktwolke mit der Maus (Drehen, Schieben, Zoomen).")
        print("3. Drücke die Taste 'Q' oder schließe das Fenster, um das Rendering zu starten!")
        print("=" * 50 + "\n")

        vis_setup = o3d.visualization.Visualizer()
        vis_setup.create_window(width=800, height=640, window_name="Setup: Positioniere Kamera, dann 'Q' drücken")

        opt_setup = vis_setup.get_render_option()
        opt_setup.background_color = np.asarray([1, 1, 1])
        opt_setup.point_size = 5.0
        opt_setup.light_on = True

        vis_setup.add_geometry(pcd)
        vis_setup.add_geometry(roi_box_lines)

        # Grobe Vorabausrichtung auf die ROI
        ctr_setup = vis_setup.get_view_control()
        ctr_setup.set_lookat(roi_center)
        ctr_setup.set_front([1.0, 1.0, 1.0])
        ctr_setup.set_up([0, 0, 1])
        ctr_setup.set_zoom(0.8)

        # Hier pausiert das Skript, bis du das Fenster schließt!
        vis_setup.run()

        # Kameraparameter speichern, BEVOR das Fenster komplett zerstört wird
        cam_params = ctr_setup.convert_to_pinhole_camera_parameters()

        # --- NEU: Tatsächliche Fensterbreite und -höhe nach dem Skalieren auslesen ---
        cam_width = cam_params.intrinsic.width
        cam_height = cam_params.intrinsic.height

        vis_setup.destroy_window()
        print(f"✅ Kamera-Perspektive gespeichert (Auflösung: {cam_width}x{cam_height}).\n")

        # ==========================================
        # PHASE 2: UNSICHTBARES RENDERING DER FRAMES
        # ==========================================
        print(f"🎬 Starte High-Quality Rendering im Hintergrund...")

        vis = o3d.visualization.Visualizer()

        # --- NEU: Das zweite Fenster bekommt exakt die Größe des ersten Fensters ---
        vis.create_window(width=cam_width, height=cam_height, visible=False)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        # ... ab hier geht dein restlicher Code normal weiter ...

        opt.point_size = 5.0
        opt.light_on = True

        vis.add_geometry(pcd)
        vis.add_geometry(roi_box_lines)

        # Zuvor gespeicherte Kamera-Einstellungen exakt anwenden
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(cam_params)

        frames_images = []
        limit = min(stream.num_frames, END_FRAME + 1)
        total_to_render = limit - START_FRAME

        for i in range(START_FRAME, limit):
            # Gesamte Wolke laden (kein Clustering mehr)
            new_pcd = stream.get_pcd(i, voxel_size=1.5)
            new_pcd.paint_uniform_color([0.1, 0.5, 0.8])

            # Normalen berechnen und für konstantes Licht ausrichten
            new_pcd.estimate_normals()
            new_pcd.orient_normals_towards_camera_location(camera_location=roi_center)

            # Punkte im Visualizer updaten
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            pcd.normals = new_pcd.normals

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Screenshot machen
            img = vis.capture_screen_float_buffer(True)
            frames_images.append((np.asarray(img) * 255).astype(np.uint8))

            # Konsolen-Update alle 15 Frames
            current_count = i - START_FRAME + 1
            if current_count % 15 == 0:
                print(f"   ⏳ Progress: {current_count} von {total_to_render} Frames fertig...")

        vis.destroy_window()

    # Video speichern
    print(f"\n💾 Speichere Video...")
    imageio.mimsave(output_video, frames_images, fps=fps, quality=10,
                    macro_block_size=16, pixelformat="yuv420p")
    print(f"✅ Video fertig: {output_video}")


if __name__ == "__main__":
    main()