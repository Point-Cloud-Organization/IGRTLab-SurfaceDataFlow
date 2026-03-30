import h5py
import pathlib


def inspect_h5_structure(file_path: pathlib.Path):
    """Scannt die interne Struktur einer HDF5 Datei."""

    if not file_path.exists():
        print(f"❌ FEHLER: Die Datei wurde nicht gefunden unter: {file_path}")
        return

    print(f"🔍 Analysiere: {file_path.name}")
    print("-" * 50)

    try:
        with h5py.File(file_path, 'r') as f:
            def print_structure(name, obj):
                # Erstellt eine Einrückung basierend auf der Hierarchie-Tiefe
                indent = "  " * name.count('/')

                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}📄 Dataset: {name}")
                    print(f"{indent}   - Shape: {obj.shape}")
                    print(f"{indent}   - Dtype: {obj.dtype}")
                    print(f"{indent}   - Compression: {obj.compression}")

                    # Metadaten/Attribute auslesen (Wichtig für Scale-Faktoren!)
                    if len(obj.attrs) > 0:
                        print(f"{indent}   - Attribute:")
                        for attr_name, attr_val in obj.attrs.items():
                            print(f"{indent}       {attr_name}: {attr_val}")

                elif isinstance(obj, h5py.Group):
                    print(f"{indent}📁 Gruppe: {name}")

            f.visititems(print_structure)

    except Exception as e:
        print(f"❌ Ein Fehler ist beim Lesen aufgetreten: {e}")


# --- KONFIGURATION ---

# 1. Pfad zur externen Festplatte anpassen
# Windows: "D:/Ordner/datei.h5"
# macOS: "/Volumes/NameDerFestplatte/Ordner/datei.h5"
pfad_zur_h5 = pathlib.Path("/Users/timjb/Documents/MUI/Radioonko/Daten/DemoData/record.h5")

if __name__ == "__main__":
    inspect_h5_structure(pfad_zur_h5)