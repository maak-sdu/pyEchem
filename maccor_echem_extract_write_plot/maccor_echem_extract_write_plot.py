import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


DPI = 300
FIGSIZE = (12,4)
LINEWIDTH = 1
FONTSIZE_LABELS = 20
FONTSIZE_TICKS = 16

VOLTAGE_MIN = 1
VOLTAGE_MAX = 3

MAJOR_TIME_INDEX = 5
MINOR_TIME_INDEX = 1
MAJOR_VOLTAGE_INDEX = 1
MINOR_VOLTAGE_INDEX = 0.25

COLORS = dict(bg_blue='#0B3C5D', bg_red='#B82601', bg_green='#1c6b0a',
              bg_lightblue='#328CC1', bg_grey='#a8b6c1', bg_yellow='#D9B310',
              bg_palered='#984B43', bg_maroon='#76323F', bg_palegreen='#626E60',
              bg_palebrown='#AB987A', bg_paleyellow='#C09F80')
COLOR = COLORS['bg_blue']


def maccor_echem_load(maccor_echem_file):
    with open(maccor_echem_file, 'r') as input_file:
        lines = input_file.readlines()
    for i,line in enumerate(lines):
        if "Rec	Cycle P	Cycle C	Step	TestTime	StepTime" in line:
            start = i + 3
    time_min_pre, time_min, time_s, time_h, voltage = [], [], [], [], []
    for i in range(start, len(lines)):
        time = float(lines[i].split()[4])
        time_min_pre.append(time)
        voltage.append(float(lines[i].split()[9]))
    for i in range(0, len(time_min_pre)):
        time_min_corrected = time_min_pre[i] - time_min_pre[0]
        time_min.append(time_min_corrected)
        time_s.append(time_min_corrected * 60)
        time_h.append(time_min_corrected / 60)
    time, time_min_pre, voltage = np.array(time), np.array(time_min_pre), np.array(voltage)
    time_min_corrected, time_min = np.array(time_min_corrected), np.array(time_min)
    time_s, time_h = np.array(time_s), np.array(time_h)
    maccor_echem_data = np.column_stack((time_h, voltage))

    return maccor_echem_data


def echem_write(echem_data, fname):
    header = "time [h]\tvoltage [V]"
    np.savetxt(f"txt/{fname}.txt", echem_data, fmt="%.10e", delimiter="\t",
               header=header, encoding="utf-8")

    return None


def echem_plot(echem_data, fname):
    time_h, voltage = echem_data[:,0], echem_data[:,1]
    fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
    plt.plot(time_h, voltage, c=COLOR, lw=LINEWIDTH)
    plt.xlim(np.amin(time_h), np.amax(time_h))
    plt.ylim(VOLTAGE_MIN, VOLTAGE_MAX)
    plt.xlabel(r"$t$ $[\mathrm{h}]$", fontsize=FONTSIZE_LABELS)
    plt.ylabel(r"$V$ $[\mathrm{V}]$", fontsize=FONTSIZE_LABELS)
    plt.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(MAJOR_TIME_INDEX))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(MINOR_TIME_INDEX))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(MAJOR_VOLTAGE_INDEX))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(MINOR_VOLTAGE_INDEX))
    plt.savefig(f"png/{fname}.png", bbox_inches="tight")
    plt.savefig(f"pdf/{fname}.pdf", bbox_inches="tight")
    plt.close()

    return None


def main():
    print(f"{80*'-'}\nFor plot settings, please see the top of this .py file.")
    data_path = Path.cwd() / "data_maccor"
    if not data_path.exists():
        data_path.mkdir()
        print(f"{80*'-'}\nA folder called '{data_path.stem}' has been created. "
              f"Please place your .txt file(s)\ncontaining maccor "
              f"electrochemical data there and rerun the code.\n{80*'-'}")
    data_files = list(data_path.glob("*.txt"))
    if len(data_files) == 0:
        print(f"{80*'-'}\nNo .txt files were found in the '{data_path.stem}' "
              f" folder. Please place your .txt\nfile(s) containing maccor "
              f"electrochemical data there and rerun the code.\n{80*'-'}")
    output_paths = ["txt", "pdf", "png"]
    for e in output_paths:
        if not (Path.cwd() / e).exists():
            (Path.cwd() / e).mkdir()
    print(f"{80*'-'}\nWorking with maccor files...")
    for f in data_files:
        print(f"\t{f.name}")
        fname = f.stem
        maccor_echem_data = maccor_echem_load(f)
        echem_write(maccor_echem_data, fname)
        echem_plot(maccor_echem_data, fname)
    print(f"Done working with files.\n{80*'-'}\nPlease see the 'txt' folder "
          f"for the extracted time [h], voltage [V] data.\nPlease see the "
          f"'pdf' and 'png' folder for plots.\n{80*'-'}")

    return None


if __name__ == "__main__":
    main()

# End of file.
