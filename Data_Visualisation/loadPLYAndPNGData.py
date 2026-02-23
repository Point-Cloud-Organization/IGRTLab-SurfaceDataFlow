import argparse
import cv2
import numpy as np
import open3d as o3d
import pathlib
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from typing import Optional
from numpy.typing import ArrayLike


class Settings:
  UNLIT = "defaultUnlit"
  LIT = "defaultLit"
  NORMALS = "normals"
  DEPTH = "depth"

  def __init__(self):
    self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
    self.bg_color = gui.Color(0.2, 0.2, 0.2)
    self.show_skybox = False
    self.show_axes = False

    self.apply_material = True  # clear to False after processing
    self._materials = {
        Settings.LIT: rendering.MaterialRecord(),
        Settings.UNLIT: rendering.MaterialRecord(),
        Settings.NORMALS: rendering.MaterialRecord(),
        Settings.DEPTH: rendering.MaterialRecord(),
    }
    self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
    self._materials[Settings.LIT].shader = Settings.LIT
    self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
    self._materials[Settings.UNLIT].shader = Settings.UNLIT
    self._materials[Settings.NORMALS].shader = Settings.NORMALS
    self._materials[Settings.DEPTH].shader = Settings.DEPTH

    self.material = self._materials[Settings.LIT]


class Stream:
  def __init__(self, folder: pathlib.Path):
    self.__folder = folder
    self.__files = self.__getDataFiles(self.__folder)
    self.__intrinsicMatrix = self.__loadIntrinsicMatrix(self.__folder.joinpath('intrinsic.txt'))
    self.__extrinsicMatrix = self.__load4x4Matrix(self.__folder.joinpath('extrinsic.txt'))
    self.__isocenterMatrix = self.__load4x4Matrix(self.__folder.joinpath('isocenter.txt'))
    self.__firstTimestamp = self.__getTimestampFromFileName(self.__files[0]['pointcloud'])
    self.__currentTimestamp = self.__firstTimestamp
    self.__currentFrame = 0
    self.__loadCurrentFrame()

  def __getTimestampFromFileName(self, file: pathlib.Path):
    return int(file.stem)

  def __getDataFiles(self, folder: pathlib.Path):
    pointCloudFiles = folder.glob("Pointclouds/*.ply")
    dataFilesPairs = []
    for pc in pointCloudFiles:
      expectedThermalFile = pc.parent.parent.joinpath("ThermalImages").joinpath(pc.with_suffix('.png').name)
      if expectedThermalFile.exists():
        dataFilesPairs.append({'pointcloud': pc, 'thermal': expectedThermalFile})
    if len(dataFilesPairs) == 0:
      print(f"Couldn't find any ply/png pair in folder: {folder}")
    return dataFilesPairs

  def __load4x4Matrix(self, path: pathlib.Path) -> Optional[ArrayLike]:
    try:
      with open(path, "r") as f:
        return np.matrix(f.readline()).reshape(4, 4)
    except Exception as ex:
      print(f"cannot read {path}: {ex}")
      pass
    return None

  def __loadIntrinsicMatrix(self, path: pathlib.Path) -> Optional[ArrayLike]:
    try:
      with open(path, "r") as f:
        for line in f.readlines():
          prefix = "Matrix: "
          if line.startswith(prefix):
            return np.matrix(line[len(prefix):]).reshape(3, 3)
    except Exception as ex:
      print(f"cannot read {path}: {ex}")
      pass
    return None

  def __loadPointcloud(self, file: pathlib.Path):
    if not file.exists():
      print(f"Error loading pointcloud: {file}")
      self.__depth = None
    self.__depth = o3d.io.read_point_cloud(str(file))
    self.__depth.transform(self.__isocenterMatrix)

  def __loadThermalImage(self, file: pathlib.Path):
    if not file.exists():
      print(f"Error loading thermal image: {file}")
      self.__thermal = None
    self.__thermal = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH)

  def __getThermalImageAsColors(self):
    thermalMin = 29314.  # ~20°C
    thermalMax = 31314.  # ~40°C
    thermal = (self.__thermal.astype(np.float32) - thermalMin) * 255. / (thermalMax - thermalMin)
    np.clip(thermal, 0, 255, out=thermal)
    thermal = thermal.astype(np.uint8)
    # open3d uses RGB[0-1] and opencv BGR[0-255], swap the channels and scale
    thermalColorMapped = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)[:, :, [2, 1, 0]] / 255.
    return thermalColorMapped

  def __project(self, depth: ArrayLike) -> ArrayLike:
    numDepthPixels = depth.shape[0]
    depthHomogeneous = np.concatenate((depth.transpose(), np.ones((1, numDepthPixels))), axis=0)
    projectionMatrix = np.concatenate((self.__intrinsicMatrix, np.zeros((3, 1))), axis=1) @ self.__extrinsicMatrix
    depthProjectedHomogenous = projectionMatrix @ depthHomogeneous
    depthProjected = depthProjectedHomogenous[:2, :] / depthProjectedHomogenous[2, :]
    return depthProjected.astype(np.int32)

  def __fuseThermalAndPointcloud(self):
    self.__thermalDepth = self.__depth
    thermalColorMapped = self.__getThermalImageAsColors()

    points2D = self.__project(np.asarray(self.__depth.points))

    thermalMappedToDepth = thermalColorMapped.reshape(thermalColorMapped.shape[0] * thermalColorMapped.shape[1], 3)[np.ravel_multi_index(
        [points2D[1, :], points2D[0, :]], self.__thermal.shape, mode="clip")].squeeze()
    self.__thermalDepth.colors = o3d.utility.Vector3dVector(thermalMappedToDepth)

  def __loadCurrentFrame(self):
    self.__loadPointcloud(self.__files[self.__currentFrame]['pointcloud'])
    self.__loadThermalImage(self.__files[self.__currentFrame]['thermal'])
    self.__fuseThermalAndPointcloud()
    self.__currentTimestamp = self.__getTimestampFromFileName(self.__files[self.__currentFrame]['pointcloud'])

  def next(self, n: int):
    self.__currentFrame = min(len(self.__files) - 1, self.__currentFrame + n)
    self.__loadCurrentFrame()
    return self.__thermalDepth

  def prev(self, n: int):
    self.__currentFrame = max(0, self.__currentFrame - n)
    self.__loadCurrentFrame()
    return self.__thermalDepth

  def goTo(self, perc: float):
    target = int(len(self.__files) * perc)
    self.__currentFrame = target
    self.__loadCurrentFrame()
    return self.__thermalDepth

  def firstTimestamp(self):
    return self.__firstTimestamp

  def currentTimestamp(self):
    return self.__currentTimestamp

  def currentPositionPercentage(self):
    return float(self.__currentFrame + 1) / float(len(self.__files))


class AppWindow:
  MENU_QUIT = 1

  def __init__(self, width: int, height: int, folder: pathlib.Path):
    self.window = gui.Application.instance.create_window("Thermal PointCloud Visualizer", width, height)
    self._scene = gui.SceneWidget()

    self._scene.scene = rendering.Open3DScene(self.window.renderer)
    self._scene.set_on_key(self._on_key)

    self.settings = Settings()

    file_menu = gui.Menu()
    file_menu.add_item("Quit", AppWindow.MENU_QUIT)

    menu = gui.Menu()
    menu.add_menu("File", file_menu)
    gui.Application.instance.menubar = menu

    self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)

    self._settings_panel = gui.Vert()

    buttons = gui.Horiz()
    prev_frame = gui.Button("Previous Frame (p)")
    prev_frame.set_on_clicked(self.prev_frame)
    next_frame = gui.Button("Next Frame (n)")
    next_frame.set_on_clicked(self.next_frame)
    buttons.add_child(prev_frame)
    buttons.add_child(next_frame)

    buttons.add_stretch()

    point_size_inc = gui.Button("Point Size + (+)")
    point_size_inc.set_on_clicked(self._on_point_size_inc)
    point_size_dec = gui.Button("Point Size - (-)")
    point_size_dec.set_on_clicked(self._on_point_size_dec)
    buttons.add_child(point_size_inc)
    buttons.add_child(point_size_dec)

    self._settings_panel.add_child(buttons)

    self._slider_update_ignore = False
    self._slider = gui.Slider(gui.Slider.Type.DOUBLE)
    self._slider.set_limits(0, 100)
    self._slider.set_on_value_changed(self._on_slider)
    self._settings_panel.add_child(self._slider)

    self._current_ts = gui.Label("")
    self._settings_panel.add_child(self._current_ts)

    self._stream_controls = [
        prev_frame,
        next_frame,
        self._slider,
    ]

    self.window.set_on_layout(self._on_layout)
    self.window.add_child(self._scene)
    self.window.add_child(self._settings_panel)

    self._current_geo = None
    self._color_overlay = None

    self.load(folder)
    self._apply_settings(True)
    self._update_current_pos_label()

  def __enter__(self):
    return self

  def __exit__(self, *args):
    return False

  def _on_layout(self, layout_context):
    settings_height = 100
    r = self.window.content_rect
    self._scene.frame = gui.Rect(r.x, r.y, r.width, r.height - settings_height)
    self._settings_panel.frame = gui.Rect(r.x, r.get_bottom() - settings_height, r.width, settings_height)

  def _apply_settings(self, apply_material: bool):
    bg_color = [
        self.settings.bg_color.red,
        self.settings.bg_color.green,
        self.settings.bg_color.blue,
        self.settings.bg_color.alpha,
    ]
    self._scene.scene.set_background(bg_color)
    self._scene.scene.show_skybox(self.settings.show_skybox)
    self._scene.scene.show_axes(self.settings.show_axes)

    if apply_material:
      self._scene.scene.update_material(self.settings.material)

  def _update_current_pos_label(self):
    pos = self._stream.currentPositionPercentage() if self._stream else 0
    self._slider_update_ignore = True
    self._slider.double_value = pos * 100

  def _update_current_ts_label(self, timestamp):
    first_timestamp = self._stream.firstTimestamp() if self._stream else 0
    self._current_ts.text = f"Timestamp: {timestamp} (+ {(timestamp - first_timestamp) / 1000:.0f}s)"

  def _on_slider(self, slider_value):
    if not self._stream:
      return
    if self._slider_update_ignore:
      self._slider_update_ignore = False
      return
    frame = self._stream.goTo(perc=slider_value / 100)
    self._show_geometry(frame, False)
    self._update_current_pos_label()
    self._update_current_ts_label(self._stream.currentTimestamp())

  def _on_key(self, evt: gui.KeyEvent):
    if evt.type == gui.KeyEvent.Type.DOWN and evt.key == 110:  # "n"
      self.next_frame()
      return gui.SceneWidget.EventCallbackResult.HANDLED
    if evt.type == gui.KeyEvent.Type.DOWN and evt.key == 270:  # "pgup"
      self.next_frame(100)
      return gui.SceneWidget.EventCallbackResult.HANDLED
    if evt.type == gui.KeyEvent.Type.DOWN and evt.key == 112:  # "p"
      self.prev_frame()
      return gui.SceneWidget.EventCallbackResult.HANDLED
    if evt.type == gui.KeyEvent.Type.DOWN and evt.key == 271:  # "pgdown"
      self.prev_frame(100)
      return gui.SceneWidget.EventCallbackResult.HANDLED
    if evt.type == gui.KeyEvent.Type.DOWN and evt.key == 93:  # "+"
      self._on_point_size_inc()
      return gui.SceneWidget.EventCallbackResult.HANDLED
    if evt.type == gui.KeyEvent.Type.DOWN and evt.key == 47:  # "-"
      self._on_point_size_dec()
      return gui.SceneWidget.EventCallbackResult.HANDLED
    return gui.SceneWidget.EventCallbackResult.IGNORED

  def _on_point_size_inc(self):
    self.settings.material.point_size += 1
    self._apply_settings(True)

  def _on_point_size_dec(self):
    self.settings.material.point_size = max(1, self.settings.material.point_size - 1)
    self._apply_settings(True)

  def _on_menu_quit(self):
    gui.Application.instance.quit()

  def _enable_stream_controls(self, enable=True):
    for control in self._stream_controls:
      control.enabled = enable

  def load(self, path: pathlib.Path):
    if not (path.is_dir() and path.joinpath("Pointclouds").is_dir() and path.joinpath("ThermalImages").is_dir()):
      raise ValueError("Loaded path is not a directory or it doesn't have the subdirectories Pointclouds and ThermalImages")
    self._enable_stream_controls(False)
    self._stream = Stream(path)
    self._enable_stream_controls()
    self.next_frame(1, True)

  def prev_frame(self, n=1, align_cam=False):
    frame = self._stream.prev(n)
    self._show_geometry(frame, align_cam)
    self._update_current_pos_label()
    if frame:
      self._update_current_ts_label(self._stream.currentTimestamp())

  def next_frame(self, n=1, align_cam=False):
    frame = self._stream.next(n)
    self._show_geometry(frame, align_cam)
    self._update_current_pos_label()
    if frame:
      self._update_current_ts_label(self._stream.currentTimestamp())

  def overlay_color(self, color: ArrayLike | None):
    self._color_overlay = color

  def _show_geometry(self, geometry: o3d.geometry.PointCloud, align_cam: bool):
    if geometry is None:
      return
    self._scene.scene.clear_geometry()
    self._current_geo = geometry
    self._scene.scene.add_geometry("__model__", self._current_geo, self.settings.material)
    if align_cam:
      bounds = self._current_geo.get_axis_aligned_bounding_box()
      self._scene.setup_camera(60, bounds, bounds.get_center())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--folder", help="Directory with the stream data.", required=True)
  args = parser.parse_args()

  gui.Application.instance.initialize()

  w = AppWindow(1024, 720, pathlib.Path(args.folder))

  gui.Application.instance.run()


if __name__ == "__main__":
  main()
