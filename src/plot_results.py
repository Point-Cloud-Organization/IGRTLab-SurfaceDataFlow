import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Pfad zur JSON-Datei
    json_path = pathlib.Path("/data/DummyDatenICP/output/icp_results.json")

    if not json_path.exists():
        print(f"Datei nicht gefunden: {json_path}")
        return

    # Daten laden
    with open(json_path, "r") as f:
        data = json.load(f)

    if not data:
        print("Die JSON-Datei ist leer.")
        return

    # Startzeitpunkt aus dem allerersten Target-File extrahieren (als Referenz für 0.0 Sekunden)
    start_time_ms = int(data[0]["target_file"].replace(".ply", ""))

    times_sec = []
    cumulative_distances = []

    # Wir starten bei Null (Einheitsmatrix) für den absoluten Verschub gegenüber Frame 0
    cumulative_matrix = np.identity(4)

    # Für den Startpunkt (0 Sekunden, 0 mm)
    times_sec.append(0.0)
    cumulative_distances.append(0.0)

    for step in data:
        # 1. Echten Zeitstempel berechnen
        current_time_ms = int(step["source_file"].replace(".ply", ""))
        time_diff_sec = (current_time_ms - start_time_ms) / 1000.0
        times_sec.append(time_diff_sec)

        # 2. Matrizen verketten (T_neu = T_schritt * T_bisher)
        step_matrix = np.array(step["transformation_matrix"])
        cumulative_matrix = step_matrix @ cumulative_matrix

        # 3. Gesamten Verschub extrahieren
        tx = cumulative_matrix[0, 3]
        ty = cumulative_matrix[1, 3]
        tz = cumulative_matrix[2, 3]

        abs_distance = np.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
        cumulative_distances.append(abs_distance)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))

    # Linie zeichnen
    plt.plot(times_sec, cumulative_distances, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)

    plt.title("Phantom Tracking: Absoluter Verschub über Zeit", fontsize=14)
    plt.xlabel("Zeit in Sekunden (ab Messbeginn)", fontsize=12)
    plt.ylabel("Gesamter Verschub in mm", fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    # Y-Achse etwas Luft nach oben geben, selbst wenn es nur Rauschen ist
    max_dist = max(cumulative_distances)
    if max_dist < 1.0:
        plt.ylim(-0.1, max_dist * 1.5 + 0.1)  # Skalierung für Rausch-Dummy-Daten

    plt.tight_layout()

    # Plot als Bild speichern (optional) und anzeigen
    plot_path = json_path.parent.joinpath("tracking_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot wurde als Bild gespeichert unter: {plot_path}")

    plt.show()


if __name__ == "__main__":
    main()