import h5py
import pathlib
import datetime


def inspect_h5_structure(file_path: pathlib.Path):
    """Scannt die interne Struktur einer HDF5 Datei und liest Timestamps aus."""

    if not file_path.exists():
        print(f"❌ FEHLER: Die Datei wurde nicht gefunden unter: {file_path}")
        return

    print(f"🔍 Analysiere: {file_path.name}")
    print("-" * 50)

    try:
        with h5py.File(file_path, 'r') as f:

            # --- 1. Allgemeine Struktur ausgeben ---
            def print_structure(name, obj):
                indent = "  " * name.count('/')
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📄 Dataset: {name}")
                    print(f"{indent}   - Shape: {obj.shape}")
                    print(f"{indent}   - Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"{indent}📁 Gruppe: {name}")

            f.visititems(print_structure)

            print("\n" + "=" * 50)
            print("🕒 TIMESTAMPS ANALYSE")
            print("=" * 50)

            # --- 2. Gezielt die Timestamps auslesen ---
            dataset_path = 'ImageStreams/ThermalDepthCamera/Timestamps'

            if dataset_path in f:
                timestamps = f[dataset_path][:]
                total_frames = len(timestamps)
                print(f"Gefundene Frames: {total_frames}\n")

                # Hilfsfunktion für die saubere Ausgabe
                def print_timestamp(index, raw_ts):
                    try:
                        if raw_ts > 10 ** 11:  # Wahrscheinlich in Millisekunden
                            readable_time = datetime.datetime.fromtimestamp(raw_ts / 1000.0).strftime(
                                '%Y-%m-%d %H:%M:%S.%f')[:-3]
                        else:  # Wahrscheinlich in Sekunden
                            readable_time = datetime.datetime.fromtimestamp(raw_ts).strftime('%Y-%m-%d %H:%M:%S')
                    except Exception:
                        readable_time = "Konnte nicht in Datum konvertiert werden"
                    print(f"  Frame {index:03d}: {raw_ts} -> {readable_time}")

                # --- Erste 10 ausgeben ---
                print("Erste Timestamps:")
                for i in range(min(10, total_frames)):
                    print_timestamp(i, timestamps[i][0])

                # --- Letzte 10 ausgeben (falls es mehr als 10 Frames gibt) ---
                if total_frames > 10:
                    if total_frames > 20:
                        print("  ...")

                    print("\nLetzte Timestamps:")
                    # Berechne den Start-Index für die letzten 10 (aber vermeide Überschneidungen mit den ersten 10)
                    start_idx = max(10, total_frames - 10)
                    for i in range(start_idx, total_frames):
                        print_timestamp(i, timestamps[i][0])

            else:
                print(f"❌ Der Pfad '{dataset_path}' existiert in dieser H5-Datei nicht.")

    except Exception as e:
        print(f"❌ Ein Fehler ist beim Lesen aufgetreten: {e}")


# --- KONFIGURATION ---
pfad_zur_h5 = pathlib.Path("/Users/timjb/Documents/MUI/Radioonko/Daten/TrackingLog.h5")

if __name__ == "__main__":
    inspect_h5_structure(pfad_zur_h5)