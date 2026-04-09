import open3d as o3d
import numpy as np
import imageio
import json
from pathlib import Path
from extractH5.h5_loader import H5PointCloudStream

# --- SETUP ---
h5_path = Path("/Users/timjb/Documents/MUI/Radioonko/Daten/1775668627720/TrackingLog.h5")
roi_path = Path("/Users/timjb/PycharmProjects/Point_Cloud/DummyDatenICP/roi_TrackingLog_124758.json")
output_video = "surf_analysis_highres.mp4"
fps = 15

# Frame-Bereich für das Shifting
START_FRAME = 40
END_FRAME = 180


def extract_main_cluster(pcd):
    labels = np.array(pcd.cluster_dbscan(eps=15.0, min_points=50))
    if len(labels) == 0 or labels.max() < 0: return pcd
    counts = np.bincount(labels[labels >= 0])
    return pcd.select_by_index(np.where(labels == counts.argmax())[0])


def main():
    with open(roi_path, "r") as f:
        roi_data = json.load(f)

    # ROI als scharfen roten Rahmen vorbereiten
    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(roi_data["min_bound"], roi_data["max_bound"])
    roi_box_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(roi_bbox)
    roi_box_lines.paint_uniform_color([1, 0, 0])

    # Visualizer Setup (800x640 für QuickTime Kompatibilität)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=640, visible=False)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # Sauberer weißer Hintergrund
    opt.point_size = 5.0  # Größere Punkte für "geschlossene" Oberfläche
    opt.light_on = True  # Beleuchtung aktivieren

    frames_images = []

    with H5PointCloudStream(h5_path) as stream:
        # Initialen Frame für Kamera-Setup
        pcd = stream.get_pcd(START_FRAME, voxel_size=1.0)
        pcd = extract_main_cluster(pcd)
        pcd.paint_uniform_color([0.1, 0.5, 0.8])
        pcd.estimate_normals()

        vis.add_geometry(pcd)
        vis.add_geometry(roi_box_lines)

        # Kamera einstellen (Diagonal: Vorne-Oben-Seitlich)
        ctr = vis.get_view_control()
        ctr.set_front([-1.2, -1.0, 1.5])  # Deine gewünschte diagonale Ansicht
        ctr.set_up([0, -1, 0])
        ctr.set_lookat(pcd.get_center())
        ctr.set_zoom(0.7)

        print(f"📸 Rendere High-Quality Video...")

        limit = min(stream.num_frames, END_FRAME + 1)
        for i in range(START_FRAME, limit):
            new_pcd = stream.get_pcd(i, voxel_size=1.0)
            new_pcd = extract_main_cluster(new_pcd)
            new_pcd.paint_uniform_color([0.1, 0.5, 0.8])

            # WICHTIG: Normalen berechnen und zur Kamera ausrichten (FIX für Grün-Stich)
            new_pcd.estimate_normals()
            new_pcd.orient_normals_towards_camera_location(camera_location=[0, 0, 0])

            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
            pcd.normals = new_pcd.normals

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Screenshot & QuickTime-Fix (YUV420p)
            img = vis.capture_screen_float_buffer(True)
            frames_images.append((np.asarray(img) * 255).astype(np.uint8))

    vis.destroy_window()

    # Video-Export
    imageio.mimsave(output_video, frames_images, fps=fps, quality=10,
                    macro_block_size=16, pixelformat="yuv420p")
    print(f"✅ Video fertig: {output_video}")


if __name__ == "__main__":
    main()