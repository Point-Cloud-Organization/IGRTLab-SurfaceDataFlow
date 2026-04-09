import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import alpha
from pathlib import Path

# 1. Zielordner festlegen (relativ zum Skript oder absoluter Pfad)
output_dir = Path("/Users/timjb/Documents/MUI/Radioonko/Poster/PC_workflow")


# --- 1. AESTHETICS & SETTINGS (Poster Style) ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['lines.linewidth'] = 2.0

import matplotlib.pyplot as plt

# --- A0 POSTER OPTIMIERUNG ---
# Größere Grundwerte für ein A0 Poster
FONT_SIZE_MAIN = 24  # Viel größer für A0
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri'],
    'font.size': FONT_SIZE_MAIN,
    'axes.titlesize': FONT_SIZE_MAIN + 4,
    'axes.labelsize': FONT_SIZE_MAIN + 2,
    'xtick.labelsize': FONT_SIZE_MAIN - 2,
    'ytick.labelsize': FONT_SIZE_MAIN - 2,
    'legend.fontsize': FONT_SIZE_MAIN - 4,
    'figure.titlesize': FONT_SIZE_MAIN + 8,
    'lines.linewidth': 4.0,      # Dickere Linien für Fernwirkung
    'axes.linewidth': 2.0,       # Dickere Rahmen
    'savefig.dpi': 600           # Maximale Schärfe für Rastergrafiken
})


# File Paths (Bitte anpassen)
ICP_FILE = "/Users/timjb/Documents/MUI/Radioonko/Poster/PC_workflow/ICP_TrackingLog_123807.csv"
MOTOR_FILE = "/Users/timjb/Documents/MUI/Radioonko/Poster/PC_workflow/ETD_ClinTest_20260408_191642.csv"


# --- 2. KINEMATICS CALCULATION ---
def calc_kinematics(h_raw, v_raw, r_raw_deg, couch_angle_deg=0.0):
    # Konstanten aus SurfKinematics
    hAxis_horizontal = 4.75 + 9.5 + 138 + 9.5 + 9.2 + 14.5
    hAxis_vertical = 14.5 + 20 + 10
    hAxis_diagonal = np.sqrt(hAxis_horizontal ** 2 + hAxis_vertical ** 2)
    radius = 130 + 21 + 75 + 46  # 46 = sliderShift

    # Lokale Kinematik
    alpha_offset = np.arctan(hAxis_vertical / hAxis_horizontal)
    pitch_rad_local = np.arcsin((hAxis_vertical + v_raw) / hAxis_diagonal) - alpha_offset
    pitch_deg_local = np.degrees(pitch_rad_local)

    rollOffset = hAxis_horizontal - np.sqrt(hAxis_diagonal ** 2 - (hAxis_vertical + v_raw) ** 2)
    y_local = -(radius + hAxis_vertical) * np.sin(pitch_rad_local) + rollOffset + (h_raw * np.cos(pitch_rad_local))
    x_local = h_raw * 0
    z_local = -radius * (1 - np.cos(pitch_rad_local)) - (h_raw * np.sin(pitch_rad_local))

    yaw_deg_local = np.degrees(np.deg2rad(r_raw_deg) * np.cos(pitch_rad_local))
    roll_deg_local = np.degrees(np.deg2rad(r_raw_deg) * np.sin(pitch_rad_local))

    # Couch Rotation (hier 0)
    couch_rad = np.deg2rad(couch_angle_deg)
    cos_g, sin_g = np.cos(couch_rad), np.sin(couch_rad)

    x_global = x_local * cos_g - y_local * sin_g
    y_global = x_local * sin_g + y_local * cos_g
    z_global = z_local

    pitch_global = pitch_deg_local * cos_g - roll_deg_local * sin_g
    roll_global = pitch_deg_local * sin_g + roll_deg_local * cos_g
    yaw_global = yaw_deg_local - couch_angle_deg

    return x_global, y_global, z_global, pitch_global, roll_global, yaw_global


# --- 3. DATA LOADING & SMART ALIGNMENT ---
def load_and_align_data():
    df_motor = pd.read_csv(MOTOR_FILE, sep=';')
    t_motor_raw = df_motor['Time_Sec'].values

    x_mot, y_mot, z_mot, pitch_mot, roll_mot, yaw_mot = calc_kinematics(
        df_motor['Pos_H'].values, df_motor['Pos_V'].values, df_motor['Pos_R'].values
    )

    # ICP Daten laden
    with open(ICP_FILE, 'r') as f:
        lines = f.readlines()
    skip_idx = next(i for i, l in enumerate(lines) if "Timestamp" in l)
    df_icp = pd.read_csv(ICP_FILE, skiprows=skip_idx)

    # Zeit normalisieren -> ICP startet bei t=0
    t_icp_raw = df_icp.iloc[:, 0].values
    t_icp = (t_icp_raw - t_icp_raw[0]) / 1000.0

    tx_icp = df_icp['Current Tx'].values
    ty_icp = df_icp['Ty'].values
    tz_icp = df_icp['Tz'].values
    rx_icp = np.degrees(df_icp['Rx'].values)
    ry_icp = np.degrees(df_icp['Ry'].values)
    rz_icp = np.degrees(df_icp['Rz'].values)
    rmse_icp = df_icp['RMSE3D'].values

    # BESSERES ALIGNMENT: Finde die aufsteigende Flanke (50% Threshold) statt dem Maximum
    # Motor: 3. Peak passiert zwischen 17 und 22 Sekunden.
    motor_mask = (t_motor_raw > 17) & (t_motor_raw < 22)
    # Finde den ersten Index, wo Motor_H über 5.0mm geht
    idx_motor_rise = np.where((df_motor['Pos_H'].values >= 5.0) & motor_mask)[0][0]
    time_motor_sync = t_motor_raw[idx_motor_rise]

    # ICP: Finde den ersten Index, wo |Ty| über 5.0mm geht
    idx_icp_rise = np.where(np.abs(ty_icp) >= 5.0)[0][0]
    time_icp_sync = t_icp[idx_icp_rise]

    # Synchronisieren
    shift = time_icp_sync - time_motor_sync
    t_motor_aligned = t_motor_raw + shift

    motor_data = (t_motor_aligned, x_mot, y_mot, z_mot, pitch_mot, roll_mot, yaw_mot)
    icp_data = (t_icp, tx_icp, ty_icp, tz_icp, rx_icp, ry_icp, rz_icp, rmse_icp)

    return motor_data, icp_data


# --- 4. PLOTTING ---
def plot_results(motor_data, icp_data):
    t_mot, x_mot, y_mot, z_mot, pitch_mot, roll_mot, yaw_mot = motor_data
    t_icp, tx_icp, ty_icp, tz_icp, rx_icp, ry_icp, rz_icp, rmse_icp = icp_data

    # Harter Cut der Achsen (Start exakt bei 0, Ende kurz nach letztem ICP Frame)
    t_min = 0.0
    t_max = t_icp[-1]

    # ==========================================
    # FIGURE 1: TRANSLATIONAL TRACKING
    # ==========================================
    fig1, axs1 = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
    fig1.canvas.manager.set_window_title('Translational Results')

    axs1[0].plot(t_mot, y_mot, '--', color='red', label='Real Phantom Shift', zorder=1, linewidth=2)
    axs1[0].plot(t_icp, -ty_icp, '-', color='#1f77b4', label='Calculated Shift in y', zorder=2, alpha=0.8, linewidth=3)
    axs1[0].set_ylabel('Longitudinal [mm]', fontweight='bold')
    axs1[0].legend(loc='upper right')
    axs1[0].grid(True, linestyle=':', alpha=0.7)
    axs1[0].set_title('Translational Tracking Results vs. Phantom Motor Postion', fontweight='bold')
    axs1[1].plot(t_mot, x_mot, '--', color='red', label='Real Phantom Shift', zorder=1, linewidth=2)
    axs1[1].plot(t_icp, tx_icp, '-', color='#ff7f0e', label='Calculated Shift in x', zorder=2, alpha=0.8, linewidth=3)
    axs1[1].set_ylabel('Lateral [mm]', fontweight='bold')
    axs1[1].legend(loc='upper right')
    axs1[1].grid(True, linestyle=':', alpha=0.7)
    axs1[1].set_ylim(-0.1, 1)


    axs1[2].plot(t_mot, z_mot, '--', color='red', label='Real Phantom Shift', zorder=1, linewidth=2)
    axs1[2].plot(t_icp, tz_icp, '-', color='#2ca02c', label='Calculated Shift in z', zorder=2, alpha=0.8, linewidth=3)
    axs1[2].set_ylabel('Vertical [mm]', fontweight='bold')
    axs1[2].legend(loc='upper right')
    axs1[2].grid(True, linestyle=':', alpha=0.7)
    axs1[2].set_ylim(-0.1, 1)

    axs1[3].plot(t_icp, rmse_icp, '-', color='#e377c2', linewidth=2)
    axs1[3].fill_between(t_icp, rmse_icp, color='#e377c2', alpha=0.2)
    axs1[3].set_xlabel('Time [s]', fontweight='bold')
    axs1[3].set_ylabel('RMSE 3D [mm]', fontweight='bold')
    axs1[3].grid(True, linestyle=':', alpha=0.7)
    axs1[3].set_xlim([t_min, t_max])
    axs1[3].set_ylim([0.7, 1.3])
    # X-Achse erzwingen
    axs1[2].set_xlim([t_min, t_max])
    fig1.tight_layout()



    plt.show(block=True)

    # Export Translationen
    fig1.savefig(output_dir / "pcPoster_Plot.svg", format='svg', bbox_inches='tight')


if __name__ == "__main__":
    motor_data, icp_data = load_and_align_data()
    plot_results(motor_data, icp_data)