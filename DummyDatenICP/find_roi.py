import open3d as o3d
import json
from pathlib import Path
from extractH5.h5_loader import H5PointCloudStream


def parse_coords(current_coords, prompt_text):
    """Hilfsfunktion für die sichere Eingabe von Koordinaten"""
    user_input = input(prompt_text).strip()
    if not user_input:
        return current_coords  # Behalte alte Werte bei Enter
    try:
        # Erlaube Leerzeichen oder Kommas als Trenner
        user_input = user_input.replace(',', ' ')
        parsed = [float(x) for x in user_input.split()]
        if len(parsed) == 3:
            return parsed
        else:
            print("⚠️ Bitte genau 3 Zahlen eingeben.")
    except ValueError:
        print("⚠️ Ungültige Eingabe. Bitte nur Zahlen eingeben.")
    return current_coords


def run_interactive_roi(h5_file: Path):
    """
    Öffnet einen Loop, um die ROI anzupassen, zu prüfen und abschließend
    unter einem dynamischen Namen zu speichern.
    Gibt die fertige o3d.geometry.AxisAlignedBoundingBox zurück.
    """
    # Startwerte
    min_bounds = [-100.0, -300.0, -50.0]
    max_bounds = [100.0, 0.0, 300.0]

    # Referenz-Frame laden
    print(f"\nLade Frame 0 aus {h5_file.name} für die ROI-Findung...")
    with H5PointCloudStream(h5_file) as stream:
        pcd = stream.get_pcd(0)

    while True:
        # ROI Box für Visualisierung erstellen
        roi = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds, max_bound=max_bounds)
        roi.color = (1, 0, 0)  # Rote Box

        # Punkte in der Box rot färben für die Vorschau (Kopie, um Original nicht dauerhaft zu färben)
        pcd_preview = o3d.geometry.PointCloud(pcd)
        pcd_cropped = pcd_preview.crop(roi)

        if not pcd_cropped.is_empty():
            pcd_cropped.paint_uniform_color([1, 0, 0])
        pcd_preview.paint_uniform_color([0.6, 0.6, 0.6])

        print("\n--- Viewer geöffnet ---")
        print("Prüfe die rote Box. Schließe das 3D-Fenster, um fortzufahren.")
        o3d.visualization.draw_geometries([pcd_preview, pcd_cropped, roi])

        # Abfrage im Terminal
        print("\nWie möchtest du fortfahren?")
        print("[j] Box ist perfekt -> Speichern & Tracking starten")
        print("[n] Box anpassen -> Neue Koordinaten eingeben")
        print("[a] Abbrechen")

        choice = input("Wahl (j/n/a): ").lower().strip()

        if choice == 'j':
            # 1. Notiz abfragen
            notiz = input("\nNotiz für den Dateinamen (Enter für keine Notiz): ").strip()

            # 2. Dateinamen generieren (z.B. roi_record_test1.json)
            base_name = h5_file.stem  # 'record' von 'record.h5'
            suffix = f"_{notiz}" if notiz else ""
            config_filename = f"roi_{base_name}{suffix}.json"

            # 3. Speichern
            calib_dir = Path("Calibration")
            calib_dir.mkdir(exist_ok=True)
            config_path = calib_dir / config_filename

            config = {
                "min_bound": min_bounds,
                "max_bound": max_bounds,
                "description": f"ROI für {h5_file.name} | Notiz: {notiz}"
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)

            print(f"✅ ROI erfolgreich in {config_path} gespeichert!")
            return roi  # Gibt die O3D BoundingBox für das Tracking zurück

        elif choice == 'n':
            print(
                "\nGib die neuen Werte ein (z.B: -100 -300 -50). Drücke einfach ENTER um den aktuellen Wert zu behalten.")
            print(f"Aktuell MIN: {min_bounds}")
            min_bounds = parse_coords(min_bounds, "Neue MIN Bounds (X Y Z): ")

            print(f"Aktuell MAX: {max_bounds}")
            max_bounds = parse_coords(max_bounds, "Neue MAX Bounds (X Y Z): ")

        elif choice == 'a':
            print("❌ Vorgang abgebrochen.")
            return None


if __name__ == "__main__":
    # Test-Aufruf, falls du das Skript doch mal alleine startest
    test_h5 = Path("/Users/timjb/PycharmProjects/Point_Cloud/DemoData/record.h5")
    run_interactive_roi(test_h5)