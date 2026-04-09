# IGRT-SurfaceDataFlow: Evaluation und Prozessierung klinischer Oberflächendaten

Dieses Repository dokumentiert die technische Basisarbeit zur Extraktion und Validierung von Oberflächen- und Thermaldaten aus modernen klinischen SGRT-Systemen. Ziel ist die Bereitstellung einer verlässlichen Pipeline zur Auswertung hochaufgelöster Punktwolken für Forschungszwecke und die klinische Qualitätssicherung.

## Projektziel und Methodik

Der Fokus dieser Arbeit liegt auf der methodischen Aufarbeitung klinischer Datenströme. Durch die Entwicklung einer dedizierten Auswerte-Pipeline wird die volle Funktionalität moderner klinischer Monitoring-Systeme technisch nachgebildet und für weiterführende Analysen zugänglich gemacht.

### Kernfunktionalitäten
* [cite_start]**HDF5-Datenextraktion**: Implementierung einer Lazy-Loading-Strategie (Klasse `H5PointCloudStream`) zum effizienten Auslesen von 3D-Datensätzen (540x720x3) direkt aus klinischen HDF5-Containern[cite: 1].
* **3D-Visualisierung & Playback**: Synchronisierte Darstellung von Oberflächengeometrie und Thermaldaten über die Zeit unter Verwendung des Open-Source-Frameworks **Open3D**.
* **DBSCAN-basierte Segmentierung**: Automatisierte Extraktion relevanter Strukturen (z. B. Phantome) aus den Rohdaten zur Reduktion von Messrauschen und Artefakten.
* **Bewegungsanalyse (ICP-Tracking)**: Anwendung des *Iterative Closest Point* Algorithmus zur Verfolgung rigider Körper mit Sub-Millimeter-Präzision.

## Klinische Anwendung: Qualitätssicherung (QA)

Die Pipeline ermöglicht eine detaillierte Validierung der Systemperformance im Rahmen der klinischen Qualitätssicherung. [cite_start]Durch den Vergleich der Tracking-Ergebnisse mit theoretischen Kinematik-Modellen (z. B. Motor-gesteuerte Phantombewegungen) können Metriken wie der RMSE3D und Score-Werte präzise bestimmt werden[cite: 1].

Dies demonstriert, dass die hier vorgestellten Open-Source-Methoden die volle Funktionalität klinischer Produkte abbilden und für unabhängige Validierungsprozesse genutzt werden können.

## Ausblick: Forschungskontext

Die saubere Extraktion und Auswertung dieser Daten bildet die Grundlage für zukünftige Forschungsprojekte. Ein zentrales Thema ist die Untersuchung von Korrelationen zwischen Oberflächenbewegung und internen Organveränderungen (Surface-to-Internal Motion Models), um langfristig digitale Zwillinge für die adaptive Strahlentherapie zu entwickeln.

---

## Technische Details

### Voraussetzungen
* Python 3.x
* [Open3D](http://www.open3d.org/)
* h5py, NumPy, Pandas

### Struktur
```text
/
├── src/
│   ├── loader/         # H5-Streaming und PLY-Verarbeitung [cite: 1]
│   ├── tracking/       # ICP-Tracking und DBSCAN-Segmentierung
│   └── visualization/  # Visualisierungstools und Plot-Generierung
├── data/               # Referenzdatensätze für die QA
└── docs/               # Dokumentation der Auswertungs-Methodik
