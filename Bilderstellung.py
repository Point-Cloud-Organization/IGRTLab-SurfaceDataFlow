import argparse
import cv2
import numpy as np
import open3d as o3d
import pathlib
import time

def draw_colorbar(image, t_min, t_max):
    # Einstellungen für die Skala
    h, w = image.shape[:2]
    bar_w = 40
    bar_h = h // 2
    x_offset = w - 150 # Etwas mehr Platz für die Zahlen
    y_offset = (h - bar_h) // 2

    # Erzeuge den Farbbalken (Jet Colormap)
    scale = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(-1, 1)
    scale_bar = cv2.applyColorMap(scale, cv2.COLORMAP_JET)
    scale_bar = cv2.repeat(scale_bar, 1, bar_w)

    # Balken ins Bild kopieren
    image[y_offset:y_offset+bar_h, x_offset:x_offset+bar_w] = scale_bar

    # Rahmen um den Balken
    cv2.rectangle(image, (x_offset, y_offset), (x_offset+bar_w, y_offset+bar_h), (255, 255, 255), 1)

    # Text-Beschriftung vorbereiten
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55
    color = (255, 255, 255)

    # Wir erstellen 5 Werte gleichmäßig verteilt
    # Falls 30000 = 30°C ist, passt /1000. Sonst anpassen!
    labels = np.linspace(t_max, t_min, 5)
    
    for i, val in enumerate(labels):
        # Berechne die Y-Position für jeden Text
        # i=0 ist oben (t_max), i=4 ist unten (t_min)
        y_pos = y_offset + int(i * (bar_h / 4))
        
        # Formatiere Text: z.B. "32.5 C"
        text = f"{val/1000:.1f} C"
        
        # Korrektur der Y-Position, damit der Text mittig zum Punkt steht
        text_y = y_pos + 5 if i == 0 else y_pos + 5
        
        cv2.putText(image, text, (x_offset + bar_w + 10, text_y), font, fs, color, 1, cv2.LINE_AA)
        
        # Kleine Hilfslinien am Balken für die Werte
        cv2.line(image, (x_offset + bar_w, y_pos), (x_offset + bar_w + 5, y_pos), color, 1)

    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required=True)
    args = parser.parse_args()
    folder = pathlib.Path(args.folder).absolute()

    pc_folder = folder / "Pointclouds"
    thermal_folder = folder / "ThermalImages"
    pcs = sorted(list(pc_folder.glob("*.ply")))
    
    def load_m(p): 
        try: return np.matrix(open(p).readline()).reshape(4,4)
        except: return np.eye(4)
    
    iso = load_m(folder / 'isocenter.txt')
    ext = load_m(folder / 'extrinsic.txt')
    
    intrinsic = np.eye(3)
    try:
        for l in open(folder / 'intrinsic.txt'):
            if l.startswith("Matrix: "): intrinsic = np.matrix(l[8:]).reshape(3,3)
    except: pass

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Export mit 5er Skala", width=1280, height=960)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.15, 0.15, 0.15]) 
    opt.point_size = 4.0 

    out_dir = folder / "Export_PNGs"
    out_dir.mkdir(exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    first = True

    print(f"Starte Export...")

    for i, pc_path in enumerate(pcs):
        thermal_path = thermal_folder / (pc_path.stem + ".png")
        if not thermal_path.exists(): continue

        temp_pcd = o3d.io.read_point_cloud(str(pc_path))
        temp_pcd.transform(iso)
        
        # Crop (Anpassen wenn oben noch Punkte schweben)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(-600, -600, -1000), 
            max_bound=(600, 600, 200) 
        )
        temp_pcd = temp_pcd.crop(bbox)
        
        therm = cv2.imread(str(thermal_path), cv2.IMREAD_ANYDEPTH)
        t_min, t_max = np.min(therm), np.max(therm)
        
        if t_max > t_min:
            t_norm = np.clip((therm.astype(np.float32) - t_min) * 255 / (t_max - t_min), 0, 255).astype(np.uint8)
        else:
            t_norm = np.zeros_like(therm, dtype=np.uint8)
            
        colors = cv2.applyColorMap(t_norm, cv2.COLORMAP_JET)[:, :, [2, 1, 0]] / 255.
        
        pts = np.asarray(temp_pcd.points)
        if len(pts) == 0: continue
        
        h = np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0)
        proj = np.concatenate((intrinsic, np.zeros((3,1))), axis=1) @ ext @ h
        uv = (proj[:2, :] / proj[2, :]).astype(np.int32)
        idx = np.ravel_multi_index([uv[1,:], uv[0,:]], therm.shape, mode="clip")
        
        pcd.points = temp_pcd.points
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3)[idx].squeeze())

        if first:
            vis.add_geometry(pcd)
            print("Kamera einstellen und 'q' drücken...")
            vis.run() 
            first = False
        else:
            vis.update_geometry(pcd)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Speichern (Start bei 1)
        fname = str(out_dir / f"frame_{i+1:04d}.png")
        vis.capture_screen_image(fname, do_render=True)
        
        # OpenCV Nachbearbeitung für die 5er Skala
        img = cv2.imread(fname)
        img = draw_colorbar(img, t_min, t_max)
        cv2.imwrite(fname, img)
        
        if (i+1) % 5 == 0:
            print(f"Fortschritt: {i+1}/{len(pcs)}")

    vis.destroy_window()
    print(f"Fertig! Alle Bilder mit detaillierter Skala sind in {out_dir}")

if __name__ == "__main__":
    main()