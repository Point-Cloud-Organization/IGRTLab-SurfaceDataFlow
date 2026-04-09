import argparse                                       # Ermöglicht das Einlesen von Kommandozeilen-Argumenten
import cv2                                            # OpenCV: Wird hier für die Bildverarbeitung (Wärmebild) genutzt
import numpy as np
import open3d as o3d                                  # Haupt-Bibliothek für 3D-Daten und Visualisierung
import pathlib                                        # Umgang mit Dateipfaden
import open3d.visualization.gui as gui                # GUI-Module für Fenster, Buttons, Slider
import open3d.visualization.rendering as rendering    # Rendering-Module für Materialien/Shader
from typing import Optional
from numpy.typing import ArrayLike



# KLASSE FÜR VISUELLE EINSTELLUNGEN 

class Settings:
  UNLIT = "defaultUnlit"    # Material ohne Schatten/Licht (flach)
  LIT = "defaultLit"        # Material auf das Lichtquellen reagiert
  NORMALS = "normals"       # Visualisierung der Oberflächen-Normalen (Vektoren)
  DEPTH = "depth"           # Visualisierung der Tiefe

  def __init__(self): 
    self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA         # Standard Maussteuerung: Kamera um Objekt drehen
    self.bg_color = gui.Color(0.2, 0.2, 0.2)                          # Hintergrundfarbe dunkelgrau
    self.show_skybox = False                                          # keine künstliche sky box
    self.show_axes = False                                            # Koordinatenachsen (x,y,z) standardmäßig aus = False

    self.apply_material = True  # clear to False after processing
    # Dictionary, das Open3D-Material-Datensätze speichert
    self._materials = {
        Settings.LIT: rendering.MaterialRecord(),
        Settings.UNLIT: rendering.MaterialRecord(),
        Settings.NORMALS: rendering.MaterialRecord(),
        Settings.DEPTH: rendering.MaterialRecord(),
    }
    # Material Eigenschaften (Farbe und Shader Typ)
    self._materials[Settings.LIT].base_color = [0.9, 0.9, 0.9, 1.0]
    self._materials[Settings.LIT].shader = Settings.LIT
    self._materials[Settings.UNLIT].base_color = [0.9, 0.9, 0.9, 1.0]
    self._materials[Settings.UNLIT].shader = Settings.UNLIT
    self._materials[Settings.NORMALS].shader = Settings.NORMALS
    self._materials[Settings.DEPTH].shader = Settings.DEPTH

    self.material = self._materials[Settings.LIT]                           # aktuell aktive Material (standardmäßig "LIT")



# KLASSE FÜR DATEN STREAM (Laden und Fusion)

class Stream:
  def __init__(self, folder: pathlib.Path):
    self.__folder = folder                                                                              # Speicherpfad merken                                                                                  
    self.__files = self.__getDataFiles(self.__folder)                                                   # Alle Datei Pfade finden
    # Matrizen aus Textdateien laden (wichtig für 3D-zu-2D Umrechnung)
    self.__intrinsicMatrix = self.__loadIntrinsicMatrix(self.__folder.joinpath('intrinsic.txt'))        
    self.__extrinsicMatrix = self.__load4x4Matrix(self.__folder.joinpath('extrinsic.txt'))
    self.__isocenterMatrix = self.__load4x4Matrix(self.__folder.joinpath('isocenter.txt'))
    # Ersten Zeitstempel aus dem Dateinamen extrahieren
    self.__firstTimestamp = self.__getTimestampFromFileName(self.__files[0]['pointcloud'])
    self.__currentTimestamp = self.__firstTimestamp                                                      
    self.__currentFrame = 0                                                                               # Starten beim ersten Bild
    self.__loadCurrentFrame()                                                                             # Daten laden

# Extrahiert Zahl aus dem Dateinamen (z.B. "12345.ply" -> 12345)
  def __getTimestampFromFileName(self, file: pathlib.Path):
    return int(file.stem)

# Sucht im Ordner nach .ply (Punktwolken) und passenden .png (Wärmebildern)
  def __getDataFiles(self, folder: pathlib.Path):
    pointCloudFiles = folder.glob("Pointclouds/*.ply")
    dataFilesPairs = []
    for pc in pointCloudFiles:
      # Pfad zum Wärmebild konstruieren: Gleicher Name wie .ply, aber .png im Nachbarordner
      expectedThermalFile = pc.parent.parent.joinpath("ThermalImages").joinpath(pc.with_suffix('.png').name)
      if expectedThermalFile.exists():
        dataFilesPairs.append({'pointcloud': pc, 'thermal': expectedThermalFile})
    if len(dataFilesPairs) == 0:
      print(f"Couldn't find any ply/png pair in folder: {folder}")
    return dataFilesPairs

# Hilfsfunktion zum Laden einer 4x4 Matrix aus einer Textdatei
  def __load4x4Matrix(self, path: pathlib.Path) -> Optional[ArrayLike]:
    try:
      with open(path, "r") as f:
        return np.matrix(f.readline()).reshape(4, 4)
    except Exception as ex:
      print(f"cannot read {path}: {ex}")
      pass
    return None

# Spezielle Ladefunktion für die Intrinsik Matrix (Kamera Eigenschaften)
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

# Punktwolke laden und mit der Isocenter Matrix im Raum ausrichten
  def __loadPointcloud(self, file: pathlib.Path):
    if not file.exists():
      print(f"Error loading pointcloud: {file}")
      self.__depth = None
      return                                                          # Schutz, falls die Datei fehlt
    self.__depth = o3d.io.read_point_cloud(str(file))                 # Datei einlesen
    self.__depth = self.__depth.voxel_down_sample(voxel_size=10)       # Anna neu hinzugefügt, Punktzahl reduzieren, 0.005 = Punkt 5mm Würfel
    self.__depth.transform(self.__isocenterMatrix)                    # Transformieren

# Wärmebild als 16-bit Rohdaten laden
  def __loadThermalImage(self, file: pathlib.Path):
    if not file.exists():
      print(f"Error loading thermal image: {file}")
      self.__thermal = None
    self.__thermal = cv2.imread(str(file), cv2.IMREAD_ANYDEPTH)

# Wandelt Roh-Wärmedaten in Farben (Colourmap JET: Blau -> Rot) um
  def __getThermalImageAsColors(self):
    thermalMin = 29314.                                                                                                       # Schwellenwert für ca 20°C, Einheit Zenti Kelvin
    thermalMax = 31314.                                                                                                       # Schwellenwert für ca 40°C
    thermal = (self.__thermal.astype(np.float32) - thermalMin) * 255. / (thermalMax - thermalMin)                             # Normalisierung der Werte auf 0.0 - 255.0
    np.clip(thermal, 0, 255, out=thermal)                                                                                     # Werte außerhalb des Bereichs kappen
    thermal = thermal.astype(np.uint8)
    # open3d uses RGB[0-1] and opencv BGR[0-255], swap the channels and scale
    thermalColorMapped = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)[:, :, [2, 1, 0]] / 255.
    return thermalColorMapped

# Berechnet die 2D Pixelkoordination für jeden 3D Punkt (Projektion)
  def __project(self, depth: ArrayLike) -> ArrayLike:
    numDepthPixels = depth.shape[0]
    # Punkte in homogene Koordinaten umwandeln (4. Zeile mit 1en)
    depthHomogeneous = np.concatenate((depth.transpose(), np.ones((1, numDepthPixels))), axis=0)
    # Projektionsmatrix erstellen: Intrinsik * Extrinsik
    projectionMatrix = np.concatenate((self.__intrinsicMatrix, np.zeros((3, 1))), axis=1) @ self.__extrinsicMatrix
    depthProjectedHomogenous = projectionMatrix @ depthHomogeneous
    # Zurück auf 2D (u,v Koordinaten) skalieren
    depthProjected = depthProjectedHomogenous[:2, :] / depthProjectedHomogenous[2, :]
    return depthProjected.astype(np.int32)

# Hauptschritt: Die Farben des Wärmebilds auf die Punkte der Wolke übertragen
  def __fuseThermalAndPointcloud(self):
    self.__thermalDepth = self.__depth
    thermalColorMapped = self.__getThermalImageAsColors()         # Farbbild holen
    points2D = self.__project(np.asarray(self.__depth.points))    # 3D -> 2D Projektion

    # Dem passenden Farbwert für jeden Punkt aus dem Bild picken (mode="clip" verhindert crash bei out-of-bounds)
    thermalMappedToDepth = thermalColorMapped.reshape(thermalColorMapped.shape[0] * thermalColorMapped.shape[1], 3)[np.ravel_multi_index(
        [points2D[1, :], points2D[0, :]], self.__thermal.shape, mode="clip")].squeeze()
    # Die Farben der Open3D Punktwolke aktualisieren
    self.__thermalDepth.colors = o3d.utility.Vector3dVector(thermalMappedToDepth)

# Lädt .ply und .png, fusion, speichert Fortschritt
  def __loadCurrentFrame(self):
    self.__loadPointcloud(self.__files[self.__currentFrame]['pointcloud'])
    self.__loadThermalImage(self.__files[self.__currentFrame]['thermal'])
    self.__fuseThermalAndPointcloud()
    self.__currentTimestamp = self.__getTimestampFromFileName(self.__files[self.__currentFrame]['pointcloud'])

# Navigation: n Frames vorwärts
  def next(self, n: int):
    self.__currentFrame = min(len(self.__files) - 1, self.__currentFrame + n)
    self.__loadCurrentFrame()
    return self.__thermalDepth

# Navigation: n Frames rückwärts
  def prev(self, n: int):
    self.__currentFrame = max(0, self.__currentFrame - n)
    self.__loadCurrentFrame()
    return self.__thermalDepth

# Navigation: Direkt zu einer Prozentstelle im Video (0.0 - 1.0)
  def goTo(self, perc: float):
    target = int(len(self.__files) * perc)
    self.__currentFrame = target
    self.__loadCurrentFrame()
    return self.__thermalDepth

# Zeitstempel und Position
  def firstTimestamp(self):
    return self.__firstTimestamp
  def currentTimestamp(self):
    return self.__currentTimestamp
  def currentPositionPercentage(self):
    return float(self.__currentFrame + 1) / float(len(self.__files))



# KLASSE FÜR GUI FENSTER

class AppWindow:
  MENU_QUIT = 1

  def __init__(self, width: int, height: int, folder: pathlib.Path):
    # Fenster erstellen
    self.window = gui.Application.instance.create_window("Thermal PointCloud Visualizer", width, height)
    self._scene = gui.SceneWidget()                                                                         # 3D Anzeigebereich
    self._scene.scene = rendering.Open3DScene(self.window.renderer)                                         
    self._scene.set_on_key(self._on_key)                                                                    # Tastatur events verknüpfen
    self.settings = Settings()

    # Menüleiste oben (File -> Quit)
    file_menu = gui.Menu()
    file_menu.add_item("Quit", AppWindow.MENU_QUIT)
    menu = gui.Menu()
    menu.add_menu("File", file_menu)
    gui.Application.instance.menubar = menu
    self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)

    # Seitenleiste/Panel für Steuerung (Vertikal angeordnet)
    self._settings_panel = gui.Vert()

    # Buttons in horizontaler Reihe
    buttons = gui.Horiz()
    prev_frame = gui.Button("Previous Frame (p)")
    prev_frame.set_on_clicked(self.prev_frame)
    next_frame = gui.Button("Next Frame (n)")
    next_frame.set_on_clicked(self.next_frame)
    buttons.add_child(prev_frame)
    buttons.add_child(next_frame)
    buttons.add_stretch()                             # Schiebt folgende Buttons nach rechts

    # Punktgröße ändern
    point_size_inc = gui.Button("Point Size + (+)")
    point_size_inc.set_on_clicked(self._on_point_size_inc)
    point_size_dec = gui.Button("Point Size - (-)")
    point_size_dec.set_on_clicked(self._on_point_size_dec)
    buttons.add_child(point_size_inc)
    buttons.add_child(point_size_dec)

    self._settings_panel.add_child(buttons)

    # Schieberegler für den zeitlichen Ablauf
    self._slider_update_ignore = False
    self._slider = gui.Slider(gui.Slider.Type.DOUBLE)
    self._slider.set_limits(0, 100)
    self._slider.set_on_value_changed(self._on_slider)
    self._settings_panel.add_child(self._slider)

    # Textlabel für den aktuellen Zeitstempel
    self._current_ts = gui.Label("")
    self._settings_panel.add_child(self._current_ts)

    # Liste der Controls zum einfachen (De-)Aktivieren
    self._stream_controls = [
        prev_frame,
        next_frame,
        self._slider,
    ]

    # Layout Verhalten 
    self.window.set_on_layout(self._on_layout)
    self.window.add_child(self._scene)
    self.window.add_child(self._settings_panel)

    self._current_geo = None
    self._color_overlay = None

    self.load(folder)                               # Daten laden starten
    self._apply_settings(True)                      # Visualisierung anwenden
    self._update_current_pos_label()                # Slider Position setzen

  # Kontext Manager Support (für "with AppWindow(...) as w:")
  def __enter__(self):
    return self

  def __exit__(self, *args):
    return False

  # Berechnet Größe des 3D Fensters vs Steuerungs Panel (unten 100 Pixel hoch)
  def _on_layout(self, layout_context):
    settings_height = 100
    r = self.window.content_rect
    self._scene.frame = gui.Rect(r.x, r.y, r.width, r.height - settings_height)
    self._settings_panel.frame = gui.Rect(r.x, r.get_bottom() - settings_height, r.width, settings_height)

  # Wendet Hintergrundfarbe und Material-Shader an
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

# Aktualisiert Schieberegler basierend auf dem Fortschritt
  def _update_current_pos_label(self):
    pos = self._stream.currentPositionPercentage() if self._stream else 0
    self._slider_update_ignore = True                                         # Verhindert Endlosschleife bei Slider Update
    self._slider.double_value = pos * 100

# Zeitstempel aktualisieren
  def _update_current_ts_label(self, timestamp):
    first_timestamp = self._stream.firstTimestamp() if self._stream else 0
    self._current_ts.text = f"Timestamp: {timestamp} (+ {(timestamp - first_timestamp) / 1000:.0f}s)"

# Reaktion auf Slider Bewegung
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

# Tastatursteuerung (n=nächster, p=vorheriger, +/-=Punktgröße)
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

# Hilfsfunktion für Punktgröße
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

# Lädt den Stream und checkt die Ordnerstruktur
  def load(self, path: pathlib.Path):
    if not (path.is_dir() and path.joinpath("Pointclouds").is_dir() and path.joinpath("ThermalImages").is_dir()):
      raise ValueError("Loaded path is not a directory or it doesn't have the subdirectories Pointclouds and ThermalImages")
    self._enable_stream_controls(False)
    self._stream = Stream(path)
    self._enable_stream_controls()
    self.next_frame(1, True)          # Ersten Frame anzeigen 

# Methoden zum Blättern
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

# Zeigt Punktwolke in 3D Szene an
  def _show_geometry(self, geometry: o3d.geometry.PointCloud, align_cam: bool):
    if geometry is None:
      return
    #self._scene.scene.clear_geometry()                                                            # Anna zum testen auskommentiert, Alte Punktwolke entfernen
    self._current_geo = geometry

#'ANNA NEU HINZUGEFÜGT ALS TEST'

    # SCHRITT 1: Prüfen, ob das Modell schon existiert
    if not self._scene.scene.has_geometry("__model__"):
        # Wenn es noch nicht da ist (beim ersten Start), normal hinzufügen
        self._scene.scene.add_geometry("__model__", self._current_geo, self.settings.material)
    else:
        # SCHRITT 2: Wenn es schon da ist, NUR die Daten (Punkte/Farben) im Speicher austauschen
        # Das ist VIEL schneller und verhindert Abstürze
        self._scene.scene.update_geometry("__model__", self._current_geo, 
                                         rendering.Scene.UPDATE_POINTS_FLAG | 
                                         rendering.Scene.UPDATE_COLORS_FLAG)
    
    # SCHRITT 3: Kamera-Ausrichtung (Die Logik bleibt, wo sie war)
    if align_cam:
        bounds = self._current_geo.get_axis_aligned_bounding_box()
        # Die Kamera wird nur auf das Objekt fokussiert
        self._scene.setup_camera(60, bounds, bounds.get_center())

    # SCHRITT 4: Der Grafikkarte sagen, dass sie das Bild jetzt neu zeichnen soll
    self._scene.force_redraw()

#'ANNA NEU HINZUGEFÜGT ALS TEST'


#'ORIGINALER ABSATZ NACH self._current_geo = geometry'

    # Neue Wolke hinzufügen mit dem Namen "__model__"
    #self._scene.scene.add_geometry("__model__", self._current_geo, self.settings.material)
    #if align_cam:                                                                                 # Kamera auf Ausdehnung der Wolke fokussieren
      #bounds = self._current_geo.get_axis_aligned_bounding_box()
      # center = bounds.get_center()                                                                # Anna neu hinzugefügt für Aufhängepunkt
      # center[2] -= 0.5                                                                            # Anna neu hinzugefügt für Aufhängepunkt
      #self._scene.setup_camera(60, bounds, bounds.get_center())                                   # self._scene.setup_camera(60, bounds, bounds.get_center()), 60 = FoV, bounds = Kasten, der alle Punkte umschließt, bounds.get... = Befehl, setze Kamera auf Mitte Kasten

#'ORIGINALER ABSATZ NACH self._current_geo = geometry'

# PROGRAMMSTART

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--folder", help="Directory with the stream data.", required=True)
  args = parser.parse_args()
  # GUI Engine initialisieren
  gui.Application.instance.initialize()
  # Hauptfenster erstellen
  w = AppWindow(1024, 720, pathlib.Path(args.folder))
  # Programm Schleife starten (wartet auf Benutzereingaben)
  gui.Application.instance.run()


if __name__ == "__main__":
  main()
