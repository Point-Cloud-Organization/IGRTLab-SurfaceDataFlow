import open3d as o3d
import numpy as np


def count_cloud_points(ply_path):
    # Datei laden
    pcd = o3d.io.read_point_cloud(ply_path)
    colors = np.asarray(pcd.colors)

    # Farben aus deiner plot_shift_poster.py
    ref_color = np.array([0.1, 0.5, 0.8])  # Poster-Blau
    shift_color = np.array([0.9, 0.4, 0.1])  # Poster-Orange

    # Masken erstellen (mit kleiner Toleranz für Rundungsfehler beim Speichern)
    ref_mask = np.all(np.isclose(colors, ref_color, atol=0.01), axis=1)
    shift_mask = np.all(np.isclose(colors, shift_color, atol=0.01), axis=1)

    num_ref = np.sum(ref_mask)
    num_shift = np.sum(shift_mask)

    print(f"📊 Analyse der Datei: {ply_path}")
    print(f"🔹 Referenz-Punkte (Blau):   {num_ref}")
    print(f"🔸 Verschobene Punkte (Orange): {num_shift}")
    print(f"Total: {len(pcd.points)} Punkte")

    return num_ref, num_shift

def main():
    count_cloud_points("poster_3d_modell.ply")

if __name__ == "__main__":
    main()