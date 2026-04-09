import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pathlib import Path

# Importiere deinen bestehenden H5-Loader
from extractH5.h5_loader import H5PointCloudStream


class AppWindow:
    def __init__(self, width: int, height: int, h5_path: Path):
        # Fenster erstellen
        self.window = gui.Application.instance.create_window("H5 Fast PointCloud Viewer", width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.set_on_key(self._on_key)

        # Einstellungen für Hintergrund und Material
        self._scene.scene.set_background([0.2, 0.2, 0.2, 1.0])
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"  # Unlit ist schneller zu rendern als Lit
        self.material.point_size = 3.0

        # --- GUI ELEMENTE (Control Panel) ---
        self._settings_panel = gui.Vert()

        # Zeile 1: Buttons und Step Size
        controls_row = gui.Horiz(0.25)

        prev_btn = gui.Button("< Prev")
        prev_btn.set_on_clicked(self.prev_frame)
        next_btn = gui.Button("Next >")
        next_btn.set_on_clicked(self.next_frame)

        step_label = gui.Label("Skip Frames:")
        self._step_input = gui.NumberEdit(gui.NumberEdit.INT)
        self._step_input.int_value = 5  # Standardmäßig 5 Frames überspringen für schnelles Sichten

        controls_row.add_child(prev_btn)
        controls_row.add_child(next_btn)
        controls_row.add_stretch()
        controls_row.add_child(step_label)
        controls_row.add_child(self._step_input)

        self._settings_panel.add_child(controls_row)

        # Zeile 2: Slider und Info Label
        slider_row = gui.Horiz(0.25)
        self._slider = gui.Slider(gui.Slider.Type.INT)
        self._slider.set_on_value_changed(self._on_slider)

        self._info_label = gui.Label("Frame: 0/0 | Timestamp: -")

        slider_row.add_child(self._slider)
        self._settings_panel.add_child(slider_row)
        self._settings_panel.add_child(self._info_label)

        # Layout Verhalten
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)
        self.window.add_child(self._settings_panel)

        # --- DATEN LADEN ---
        self.current_frame = 0
        self._current_geo = None

        print(f"Lade H5 Datei: {h5_path}...")
        self.stream = H5PointCloudStream(h5_path)

        # Slider Limits setzen
        self._slider.set_limits(0, self.stream.num_frames - 1)

        # Ersten Frame laden
        self._load_and_show_frame(align_cam=True)

    def _on_layout(self, layout_context):
        # Panel unten verankern (Höhe ca. 90 Pixel)
        settings_height = 90
        r = self.window.content_rect
        self._scene.frame = gui.Rect(r.x, r.y, r.width, r.height - settings_height)
        self._settings_panel.frame = gui.Rect(r.x, r.get_bottom() - settings_height, r.width, settings_height)

    def _on_key(self, evt: gui.KeyEvent):
        # Steuerung mit Tastatur (Pfeil Rechts/Links oder N/P)
        if evt.type == gui.KeyEvent.Type.DOWN:
            if evt.key == gui.KeyName.RIGHT or evt.key == 110:  # Pfeil Rechts oder 'n'
                self.next_frame()
                return gui.SceneWidget.EventCallbackResult.HANDLED
            elif evt.key == gui.KeyName.LEFT or evt.key == 112:  # Pfeil Links oder 'p'
                self.prev_frame()
                return gui.SceneWidget.EventCallbackResult.HANDLED
        return gui.SceneWidget.EventCallbackResult.IGNORED

    def _on_slider(self, slider_value):
        self.current_frame = int(slider_value)
        self._load_and_show_frame(align_cam=False)

    def prev_frame(self):
        step = self._step_input.int_value
        self.current_frame = max(0, self.current_frame - step)
        self._slider.int_value = self.current_frame
        self._load_and_show_frame(align_cam=False)

    def next_frame(self):
        step = self._step_input.int_value
        self.current_frame = min(self.stream.num_frames - 1, self.current_frame + step)
        self._slider.int_value = self.current_frame
        self._load_and_show_frame(align_cam=False)

    def _load_and_show_frame(self, align_cam: bool):
        # 1. Daten abrufen (mit Voxel-Downsampling für maximale Geschwindigkeit)
        pcd = self.stream.get_pcd(self.current_frame, voxel_size=1.5)

        # Wolke einfärben (z.B. nach Z-Höhe oder einfarbig)
        pcd.paint_uniform_color([0.5, 0.7, 0.9])  # Hellblau

        self._current_geo = pcd

        # 2. Hocheffizientes Rendering (Robuster Fallback statt update_geometry)
        if self._scene.scene.has_geometry("__model__"):
            self._scene.scene.remove_geometry("__model__")

        self._scene.scene.add_geometry("__model__", self._current_geo, self.material)

        # 3. Kameraausrichtung (nur beim ersten Frame)
        if align_cam and not self._current_geo.is_empty():
            bounds = self._current_geo.get_axis_aligned_bounding_box()
            self._scene.setup_camera(60, bounds, bounds.get_center())

        self._scene.force_redraw()

        # 4. Info-Label aktualisieren
        depth_ts, _ = self.stream.get_timestamps(self.current_frame)
        self._info_label.text = f"Frame: {self.current_frame}/{self.stream.num_frames - 1} | Timestamp: {depth_ts}"

def main():

    gui.Application.instance.initialize()

    # H5-Pfad aus Argument lesen
    h5_path = Path("/Users/timjb/Documents/MUI/Radioonko/Daten/1775668627720/TrackingLog.h5")
    if not h5_path.exists():
        print(f"Fehler: Datei {h5_path} nicht gefunden.")
        return

    w = AppWindow(1280, 800, h5_path)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()