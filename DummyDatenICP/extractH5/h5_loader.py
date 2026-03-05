import h5py
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple


class H5PointCloudStream:
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        # libver='latest' optimiert die Lese-Performance bei modernen H5-Dateien
        self._file = h5py.File(h5_path, 'r', libver='latest')

        # 1. Kalibrierung laden
        self.isocenter_matrix = self._file['AdditionalInformation/Calibration/IsocenterCalibration'][()]

        # 2. Referenzen auf die fetten Datasets (werden hier NICHT in den RAM geladen)
        self.depth_ds = self._file['ImageStreams/DepthCamera/DepthData']
        self.thermal_ds = self._file['ImageStreams/ThermalCamera/ThermalData']

        # 3. PERFORMANCE-BOOST: Kleine, gzip-komprimierte Daten sofort komplett in den RAM laden
        # Das [:] Slicing zieht das gesamte Dataset sofort als schnelles NumPy-Array in den Speicher
        self.depth_timestamps = self._file['ImageStreams/DepthCamera/Timestamps'][:]
        self.thermal_timestamps = self._file['ImageStreams/ThermalCamera/Timestamps'][:]

        self.num_frames = self.depth_ds.shape[0]

    def get_pcd(self,
                index: int,
                roi_bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = None,
                voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """
        Lädt die 3D-Daten hochoptimiert.
        Erlaubt direktes Zuschneiden (roi_bbox) und Downsampling (voxel_size),
        um C++-Konvertierungskosten in Open3D zu sparen.
        """
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Index {index} außerhalb des Bereichs (0 bis {self.num_frames - 1})")

        # Auslesen und flach machen
        raw_points = self.depth_ds[index].reshape(-1, 3)

        # PERFORMANCE-BOOST: Wir checken nur die Z-Koordinate (Index 2).
        # Das spart extrem viel Zeit im Vergleich zum Check über alle Achsen.
        z_coords = raw_points[:, 2]
        valid_mask = np.isfinite(z_coords) & (z_coords != 0.0)
        clean_points = raw_points[valid_mask]

        # Open3D Konvertierung (teuerster Schritt, profitiert massiv davon, wenn clean_points klein ist)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(clean_points)

        # Transformation direkt anwenden
        pcd.transform(self.isocenter_matrix)

        # Geometrie optimieren BEVOR sie ins Hauptskript geht
        if roi_bbox is not None:
            pcd = pcd.crop(roi_bbox)
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        return pcd

    def get_thermal(self, index: int) -> np.ndarray:
        """Lädt NUR das Wärmebild für den Index."""
        return self.thermal_ds[index].squeeze()

    def get_timestamps(self, index: int) -> Tuple[int, int]:
        """Liest super flashlike verdammt blitzschnell aus dem RAM-Cache."""
        # Wir greifen nun auf das RAM-Array zu, nicht mehr auf das H5-Dataset!
        return self.depth_timestamps[index][0], self.thermal_timestamps[index][0]

    # Ein Pythonischer Generator für blitzschnelles Loopen
    def iter_frames(self, step: int = 1, roi_bbox=None, voxel_size=None):
        """Erlaubt elegantes Loopen: for pcd in stream.iter_frames():"""
        for i in range(0, self.num_frames, step):
            yield i, self.get_pcd(i, roi_bbox, voxel_size)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()