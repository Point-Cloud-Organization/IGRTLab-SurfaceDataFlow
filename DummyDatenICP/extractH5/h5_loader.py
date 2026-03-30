import os
import hdf5plugin
import h5py
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple

# HDF5 Plugin Pfad setzen
try:
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.plugin_path
except AttributeError:
    os.environ["HDF5_PLUGIN_PATH"] = os.path.join(os.path.dirname(hdf5plugin.__file__), 'plugins')


class H5PointCloudStream:
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self._file = h5py.File(h5_path, 'r', libver='latest')

        # 1. Kalibrierung laden
        self.isocenter_matrix = self._file['AdditionalInformation/Calibration/IsocenterCalibration'][()]

        # 2. Dynamische Pfad-Erkennung für neue und alte Struktur
        if 'ImageStreams/ThermalDepthCamera' in self._file:
            # NEUE STRUKTUR: Alles in einem Ordner
            print(f"DEBUG: Neue H5-Struktur erkannt (ThermalDepthCamera)")
            base_depth = 'ImageStreams/ThermalDepthCamera'
            base_thermal = 'ImageStreams/ThermalDepthCamera'
        else:
            # ALTE STRUKTUR: Getrennte Ordner
            print(f"DEBUG: Alte H5-Struktur erkannt (DepthCamera/ThermalCamera)")
            base_depth = 'ImageStreams/DepthCamera'
            base_thermal = 'ImageStreams/ThermalCamera'

        # Datasets zuweisen
        self.depth_ds = self._file[f'{base_depth}/DepthData']
        self.thermal_ds = self._file[f'{base_thermal}/ThermalData']

        # 3. Zeitstempel laden (Caching im RAM)
        # In der neuen Struktur gibt es oft nur einen gemeinsamen Timestamps-Eintrag
        self.depth_timestamps = self._file[f'{base_depth}/Timestamps'][:]

        if base_depth == base_thermal:
            # In der neuen kombinierten Struktur nutzen wir dieselben Timestamps
            self.thermal_timestamps = self.depth_timestamps
        else:
            # In der alten Struktur laden wir den separaten thermalen Timestamp
            self.thermal_timestamps = self._file[f'{base_thermal}/Timestamps'][:]

        self.num_frames = self.depth_ds.shape[0]

    def get_pcd(self,
                index: int,
                roi_bbox: Optional[o3d.geometry.AxisAlignedBoundingBox] = None,
                voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        if index < 0 or index >= self.num_frames:
            raise IndexError(f"Index {index} außerhalb des Bereichs")

        raw_points = self.depth_ds[index].reshape(-1, 3)
        z_coords = raw_points[:, 2]
        valid_mask = np.isfinite(z_coords) & (z_coords != 0.0)
        clean_points = raw_points[valid_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(clean_points)
        pcd.transform(self.isocenter_matrix)

        if roi_bbox is not None:
            pcd = pcd.crop(roi_bbox)
        if voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        return pcd

    def get_thermal(self, index: int) -> np.ndarray:
        return self.thermal_ds[index].squeeze()

    def get_timestamps(self, index: int) -> Tuple[int, int]:
        # Gibt (depth_ts, thermal_ts) zurück
        return self.depth_timestamps[index][0], self.thermal_timestamps[index][0]

    def iter_frames(self, step: int = 1, roi_bbox=None, voxel_size=None):
        for i in range(0, self.num_frames, step):
            yield i, self.get_pcd(i, roi_bbox, voxel_size)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()