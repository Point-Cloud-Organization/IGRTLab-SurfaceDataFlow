import cv2
import pathlib
import os

def create_video(image_folder, video_name, fps=10):
    images = sorted(list(pathlib.Path(image_folder).glob("frame_*.png")))
    
    if not images:
        print("Keine Bilder gefunden!")
        return

    # Erstes Bild laden, um die Größe zu bestimmen
    frame = cv2.imread(str(images[0]))
    height, width, layers = frame.shape

    # VideoWriter definieren (MP4 Format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    print(f"Erstelle Video aus {len(images)} Bildern...")
    for image_path in images:
        video.write(cv2.imread(str(image_path)))

    video.release()
    print(f"Fertig! Das Video wurde als '{video_name}' gespeichert.")

# Ausführung
folder_path = "./Export_PNGs_Seite" # Dein Ordner mit den Bildern
create_video(folder_path, "Thermal_Sequence_Seite_822.mp4", fps=10)