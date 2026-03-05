🛠 Technical Overview: Point Cloud Pipeline & Visualization
The core of this toolkit is the high-efficiency processing of massive 3D datasets directly from HDF5 containers. By implementing a Lazy Loading strategy within the H5PointCloudStream class, frames (Shape: 540x720x3) are only read from the disk upon explicit request, preserving system memory even during long-term measurements exceeding 6,000 frames.

Optimized Extraction: The loader filters invalid measurement points using a high-performance check of the Z-coordinate column for NaN or zero values before converting the data into the Open3D format.

Automated Alignment: Every point cloud is automatically oriented in space during the loading process using the embedded 4x4 IsocenterCalibration matrix to ensure a consistent coordinate base for the ICP algorithm.

Performance & ROI: To accelerate computations, the loader provides integrated hooks for Region-of-Interest (ROI) cropping and Voxel Downsampling, significantly reducing the data volume per frame for real-time tracking.

QA Validation: Tracking quality is validated via exported metrics such as RMSE3D and Score in CSV logs, ensuring sub-millimeter precision for clinical quality assurance.
