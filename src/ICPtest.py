import pathlib
import json
import numpy as np
import open3d as o3d
from typing import Optional
from numpy.typing import ArrayLike


# --- HILFSFUNKTIONEN FÜR MATRIZEN ---

def load_4x4_matrix(path: pathlib.Path) -> Optional[ArrayLike]:
    try:
        with open(path, "r") as f:
            return np.matrix(f.readline()).reshape(4, 4)
    except Exception as ex:
        print(f"Cannot read {path}: {ex}")
        return None


def load_intrinsic_matrix(path: pathlib.Path) -> Optional[ArrayLike]:
    try:
        with open(path, "r") as f:
            for line in f.readlines():
                prefix = "Matrix: "
                if line.startswith(prefix):
                    return np.matrix(line[len(prefix):]).reshape(3, 3)
    except Exception as ex:
        print(f"Cannot read {path}: {ex}")
        return None


# --- VORVERARBEITUNG DER PUNKTWOLKEN ---

def preprocess_pointcloud(file_path: pathlib.Path, isocenter_matrix: ArrayLike) -> o3d.geometry.PointCloud:
    # 1. Laden
    pcd = o3d.io.read_point_cloud(str(file_path))

    # 2. Ausrichten (Isocenter)
    if isocenter_matrix is not None:
        pcd.transform(isocenter_matrix)

    # 3. Voxel Downsampling (2mm Würfel für gute Performance)
    pcd = pcd.voxel_down_sample(voxel_size=2.0)

    # 4. Normalen berechnen (Zwingend notwendig für Point-to-Plane ICP)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    return pcd


# --- HAUPTPROGRAMM ---

def main():
    # 1. Hauptpfad definieren
    folder = pathlib.Path("/data/DummyDatenICP")  # <-- Hier deinen Ordnernamen einsetzen

    if not folder.exists():
        print(f"Ordner {folder} nicht gefunden!")
        return

    # Output Ordner erstellen
    output_folder = folder.joinpath("output")
    output_folder.mkdir(parents=True, exist_ok=True)

    # 2. Matrizen aus dem "Calibration" Unterordner laden
    isocenter_path = folder.joinpath('Calibration', 'isocenter.txt')
    isocenter = load_4x4_matrix(isocenter_path)

    if isocenter is None:
        print(f"Warnung: {isocenter_path} konnte nicht geladen werden.")

    # 3. Alle .ply Dateien im "PointCloud" Ordner finden
    # Wichtig: Achte exakt auf die Groß-/Kleinschreibung des Ordnernamens!
    ply_files = sorted(list(folder.glob("PointCloud/*.ply")), key=lambda x: int(x.stem))

    if len(ply_files) < 2:
        print(f"Nicht genug Punktwolken in {folder.joinpath('PointCloud')} gefunden.")
        return

    print(f"{len(ply_files)} Punktwolken gefunden. Starte Vorverarbeitung...")

    # Liste, in der wir die JSON-Daten sammeln
    results_data = []

    # 4. Ersten Frame laden (Target)
    target_pcd = preprocess_pointcloud(ply_files[0], isocenter)

    # 5. Schleife über die restlichen Frames
    for i in range(1, len(ply_files)):
        print(f"--- Berechne Verschub: Frame {i - 1} -> Frame {i} ---")

        # Aktuellen Frame laden (Source)
        source_pcd = preprocess_pointcloud(ply_files[i], isocenter)

        # ICP Parameter
        distance_threshold = 20.0
        trans_init = np.identity(4)

        # Point-to-Plane ICP ausführen
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, distance_threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
        )

        # Verschub extrahieren
        transformation_matrix = np.asarray(reg_p2p.transformation)
        tx = transformation_matrix[0, 3]
        ty = transformation_matrix[1, 3]
        tz = transformation_matrix[2, 3]

        # Euklidische Distanz berechnen
        distance = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)

        print(f"-> Absoluter Verschub: {distance:.5f} mm\n")

        # Daten für das JSON-Format vorbereiten
        # .tolist() ist wichtig, da json.dump keine numpy-Arrays verarbeiten kann
        frame_result = {
            "source_frame_index": i,
            "target_frame_index": i - 1,
            "source_file": ply_files[i].name,
            "target_file": ply_files[i - 1].name,
            "transformation_matrix": transformation_matrix.tolist(),
            "translation_mm": {
                "tx": float(tx),
                "ty": float(ty),
                "tz": float(tz)
            },
            "absolute_distance_mm": float(distance)
        }
        results_data.append(frame_result)

        # Nächster Durchlauf: Jetzige Source wird zum neuen Target
        target_pcd = source_pcd

    # 6. Ergebnisse in eine JSON-Datei schreiben
    output_file = output_folder.joinpath("icp_results.json")
    with open(output_file, "w") as json_file:
        json.dump(results_data, json_file, indent=4)

    print(f"Fertig! Ergebnisse wurden erfolgreich in gespeichert: {output_file}")


if __name__ == "__main__":
    main()