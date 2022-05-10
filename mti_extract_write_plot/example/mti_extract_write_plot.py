import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from datetime import datetime
try:
    PLOT_STYLE = True
    from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style
except ImportError:
    PLOT_STYLE = None
from scipy.constants import physical_constants


# Inputs for loading MTI data from .txt file
HEADER = 5
VOLTAGE_INDEX = 2
CURRENT_INDEX = 3
DATE_INDEX = -2
TIME_INDEX = -1

# Inputs to calculate amount of working ion transferred
WORKING_ION_CHARGE = 1
WORKING_ION_START_VALUE = 0
MOLAR_MASS = 79.866
MASS = 0.6 * 11.276 * 10**-3

# Inputs for plots
DPI = 600
FIGSIZE = (12,4)
FONTSIZE_LABELS = 20
FONTSIZE_TICKS = 14
XLABEL = "$x$ in Li$_{x}$TiO$_{2}$"
TIMELABEL = "$t$ $[\mathrm{h}]$"
VOLTAGELABEL = "$E_{\mathrm{we}}$ vs. Li/Li$^{+}$"
MAJOR_TICK_INDEX_TIME = 5
MAJOR_TICK_INDEX_VOLTAGE = 0.5
MAJOR_TICK_INDEX_X = 0.1
VOLTAGE_LIMITS = True
VOLTAGE_MIN = 1
VOLTAGE_MAX = 3
BREAKFACTOR_X = 0.023
BREAKFACTOR_Y = 0.02
TOLERANCE_FACTOR = 10**2


def mti_to_dict_extract(file):
    with file.open(mode="r") as f:
        lines = f.readlines()
    start = None
    voltage, current, date, time, date_time = [], [], [], [], []
    for i in range(HEADER, len(lines)):
        line_split = lines[i].split()
        if line_split[1] == "CC_DChg":
            if isinstance(start, type(None)):
                start = i + 1
                break
        elif line_split[1] == "CC_Chg":
            if isinstance(start, type(None)):
                start = i
                break
    mode_change_indices = []
    for i in range(start, len(lines)):
        line_split = lines[i].split()
        if line_split[1] == "Rest":
            mode_change_indices.append(i)
        elif line_split[1] == "CC_DChg":
            mode_change_indices.append(i)
        elif line_split[1] == "CC_Chg":
            mode_change_indices.append(i)
    for i in range(start, len(lines)):
        if not i in mode_change_indices:
            line_split = lines[i].split()
            if i == start:
                cols = len(line_split)
            if len(line_split) == cols:
                voltage.append(float(line_split[VOLTAGE_INDEX]))
                current.append(float(line_split[CURRENT_INDEX]))
                date.append(line_split[DATE_INDEX])
                time.append(line_split[TIME_INDEX])
    for i in range(len(date)):
        date_split = date[i].split("-")
        time_split = time[i].split(":")
        date_time.append(datetime(int(date_split[0]),
                                  int(date_split[1]),
                                  int(date_split[2]),
                                  int(time_split[0]),
                                  int(time_split[1]),
                                  int(time_split[2])
                                  ).timestamp()
                        )
    voltage_v = np.array(voltage) / 10**3
    current_a = np.array(current) / 10**3
    date_time = np.array(date_time) / 60**2
    time_h = (date_time - date_time[0])
    d = {"time_h": time_h, "voltage_v": voltage_v, "current_a" : current_a}

    return d


def x_from_dict_calcualte(d):
    time_h, current_a = d["time_h"], d["current_a"]
    x = [WORKING_ION_START_VALUE]
    n = MASS / MOLAR_MASS
    f = physical_constants["Faraday constant"][0]
    for i in range(1, len(time_h)):
        delta_q =  - current_a[i] * (time_h[i] - time_h[i-1]) * 60**2
        delta_x = delta_q / (n * f)
        x.append(x[i-1] + delta_x)
    change_indices = [i for i in range(1, len(current_a))
                      if current_a[i] != 0
                      and current_a[i] * current_a[i-1] <= 0]
    d["x"], d["change_indices"] = np.array(x), np.array(change_indices)

    return d


def dict_plot(d, output_paths):
    time_h, voltage_v, x = d["time_h"], d["voltage_v"], d["x"]
    change_indices = d["change_indices"]
    time_min, time_max = np.amin(time_h), np.amax(time_h)
    time_range = time_max - time_min
    voltage_min, voltage_max = np.amin(voltage_v), np.amax(voltage_v)
    voltage_range = voltage_max - voltage_min
    x_min, x_max = np.amin(x), np.amax(x)
    x_range = x_max - x_min
    t_changes = [time_h[e] for e in change_indices]
    t_changes_labels = [f"{x[e]:.3f}" for e in change_indices]
    xticks_labels = [f"{e:.1f}" for e in np.arange(0, 0.8, 0.1)]
    xticks_labels.append(t_changes_labels[0])
    for e in np.arange(0.7, 0.3, -0.1):
        xticks_labels.append(f"{e:.1f}")
    xticks_labels.append(t_changes_labels[1])
    for e in np.arange(0.4, 0.5, 0.1):
        xticks_labels.append(f"{e:.1f}")
    t_xticks = np.array([])
    j = 0
    for i in range(0, len(x)):
        if np.isclose(np.array(xticks_labels[j], dtype=float),
                      x[i],
                      atol=abs(x[0] - x[1]) * TOLERANCE_FACTOR
                      ):
            t_xticks = np.append(t_xticks, time_h[i])
            j += 1
            if j == len(xticks_labels):
                break
    if not isinstance(PLOT_STYLE, type(None)):
        plt.style.use(bg_mpl_style)
    fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
    ax.plot(time_h, voltage_v)
    ax.set_xlim(time_min, time_max)
    if not VOLTAGE_LIMITS is True:
        ax.set_ylim(voltage_min, voltage_max)
        ax.set_ylim(voltage_min, voltage_max)
    else:
        ax.set_ylim(VOLTAGE_MIN, VOLTAGE_MAX)
        ax.set_ylim(VOLTAGE_MIN, VOLTAGE_MAX)
    ax.set_xlabel(TIMELABEL, fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(VOLTAGELABEL, fontsize=FONTSIZE_LABELS)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_TIME))
    ax.xaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_TIME / 5))
    ax.yaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE))
    ax.yaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE / 5))
    for p in output_paths:
        plt.savefig(f"{p}/mti_t_v_plot.{p}", bbox_inches="tight")
    plt.close()
    fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
    ax.plot(x, voltage_v)
    ax.set_xlim(x_min, x_max)
    if not VOLTAGE_LIMITS is True:
        ax.set_ylim(voltage_min, voltage_max)
        ax.set_ylim(voltage_min, voltage_max)
    else:
        ax.set_ylim(VOLTAGE_MIN, VOLTAGE_MAX)
        ax.set_ylim(VOLTAGE_MIN, VOLTAGE_MAX)
    ax.set_xlabel(XLABEL, fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(VOLTAGELABEL, fontsize=FONTSIZE_LABELS)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_X))
    ax.xaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_X / 5))
    ax.yaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE))
    ax.yaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE / 5))
    for p in output_paths:
        plt.savefig(f"{p}/mti_x_v_plot.{p}", bbox_inches="tight")
    plt.close()
    fig = plt.figure(dpi=DPI, figsize=FIGSIZE)
    ax0 = fig.add_subplot(111)
    ax1 = ax0.twiny()
    ax1.plot(time_h, voltage_v)
    ax1.set_xlim(time_min, time_max)
    ax0.set_xlim(time_min, time_max)
    if not VOLTAGE_LIMITS is True:
        ax1.set_ylim(voltage_min, voltage_max)
        ax0.set_ylim(voltage_min, voltage_max)
    else:
        ax1.set_ylim(VOLTAGE_MIN, VOLTAGE_MAX)
        ax0.set_ylim(VOLTAGE_MIN, VOLTAGE_MAX)
    ax0.set_xticks(t_xticks)
    ax0.set_xticklabels(xticks_labels)
    ax0.set_xlabel(XLABEL, fontsize=FONTSIZE_LABELS)
    ax1.set_xlabel(TIMELABEL, fontsize=FONTSIZE_LABELS)
    ax0.set_ylabel(VOLTAGELABEL, fontsize=FONTSIZE_LABELS)
    ax1.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_TIME))
    ax1.xaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_TIME / 5))
    ax1.yaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE))
    ax1.yaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE / 5))
    ax0.xaxis.set_tick_params(labelsize=FONTSIZE_TICKS)
    ax1.xaxis.set_tick_params(labelsize=FONTSIZE_TICKS)
    ax0.yaxis.set_tick_params(labelsize=FONTSIZE_TICKS)
    for i in range(len(t_changes)):
        plt.text(t_changes[i] - BREAKFACTOR_X * time_range,
                 voltage_min - BREAKFACTOR_Y * voltage_range,
                 "|",
                 rotation=45)
    for p in output_paths:
        plt.savefig(f"{p}/mti_x_t_v_plot.{p}", bbox_inches="tight")
    plt.close()

    return None


def main():
    data_path = Path.cwd() / "data"
    if not data_path.exists():
        data_path.mkdir()
        print(f"{80*'-'}\nA folder called 'data' has been created.\nPlease "
              f"place your data (.txt) files containing data from the MTI "
              f"potentiostat\nthere and rerun the program.\n{80*'-'}")
        sys.exit()
    data_files = list(data_path.glob("*.txt"))
    if len(data_files) == 0:
        print(f"{80*'-'}\nNo .txt files were found in the 'data' folder.\n"
              f"Please place your data (.txt) files containing data from the "
              f"MTI potentiostat\nthere and rerun the program.\n{80*'-'}")
        sys.exit()
    txt_path = Path.cwd() / "txt"
    if not txt_path.exists():
        txt_path.mkdir()
    header = "t [h]\tV [V]\ti [A]"
    output_paths = ["pdf", "png", "svg"]
    for p in output_paths:
        if not (Path.cwd() / p).exists():
            (Path.cwd() / p).mkdir()
    for file in data_files:
        d = mti_to_dict_extract(file)
        np.savetxt(txt_path / file.name,
                   np.column_stack((d["time_h"],
                                    d["voltage_v"],
                                    d["current_a"]
                                    )),
                   header=header,
                   delimiter="\t",
                   encoding="utf-8",
                   fmt="%.4e"
                   )
        d = x_from_dict_calcualte(d)
        dict_plot(d, output_paths)

    return None


if __name__ == "__main__":
    main()

# End of file.
