import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style
    STYLE = True
except ImportError:
    STYLE = False
    COLORS = dict(bg_blue='#0B3C5D', bg_red='#B82601', bg_green='#1c6b0a',
                  bg_lightblue='#328CC1', bg_grey='#a8b6c1', bg_yellow='#D9B310',
                  bg_palered='#984B43', bg_maroon='#76323F', bg_palegreen='#626E60',
                  bg_palebrown='#AB987A', bg_paleyellow='#C09F80')
    COLOR = COLORS['bg_blue']

DPI=600
FIGSIZE = (8,4)
FONTSIZE_LABELS = 20
FONTSIZE_TICKS = 14
LINEWIDTH = 1

def comma_to_dot(txt_files, comma_to_dot_path):
    print(f"{80*'-'}\nConverting commas to dots...")
    for file in txt_files:
        print(f"\t{file.name}")
        with file.open() as f:
            txt = f.read().replace(",", ".")
        output_path_file = comma_to_dot_path / file.name
        with open(output_path_file, mode="w") as o:
            o.write(txt)
    print(f"Done converting comma to dots.\nFiles saved to the "
          f"'comma_to_dot' folder.")

    return None


def ec_lab_extractor(txt_files_dotted, output_path, plotpaths):
    for file in txt_files_dotted:
        print(f"{80*'-'}\nFile:\t{file.name}")
        with file.open() as f:
            df = pd.read_csv(f, delimiter="\t")
        print("\n\tThe following columns are available:")
        keys = list(k for k in df.keys() if not "unnamed" in k.lower())
        for e in enumerate(keys):
            print(f"\t\t{e[0]}\t{e[1]}")
        x_desire = int(input("\n\tPlease indicate the number for the desired x quantity: "))
        y_desire = int(input("\tPlease indicate the number for the desired y quantity: "))
        x_desire_key = keys[x_desire]
        y_desire_key = keys[y_desire]
        x_values, y_values = df[x_desire_key].to_numpy(), df[y_desire_key].to_numpy()
        if x_desire_key == "time/s":
            x_values = x_values / 60**2
        rest_desire = input("\n\tDo you want to remove any initial resting from the data? (y/n): ")
        print(f"\n\tExtracting data...")
        if rest_desire == "y":
            if "x" in keys:
                x = df["x"].to_numpy()
                for i in range(1, len(x)):
                    if x[i-1] == x[0] and x[i] != x[i-1]:
                        x_start_index = i
                x_values, y_values = x_values[x_start_index:], y_values[x_start_index:]
            elif "<i>/mA" in keys:
                current = df["<i>/mA"].to_numpy()
                for i in range(1, len(current)):
                    if current[i-1] == current[0] and current[i] != current[i-1]:
                        current_start_index = i
                x_values, y_values = x_values[current_start_index:], y_values[current_start_index:]
            x_values = x_values - x_values[0]
        else:
            pass
        if x_desire_key == "time/s":
            x_name = "t"
        elif x_desire_key == "Ewe/V":
            x_name = "V"
        elif "x" == x_desire_key:
            x_name = "x"
        elif "<i>/mA" == x_desire_key:
            x_name = "i"
        elif "Q discharge/mA.h" == x_desire_key:
            x_name = "Qdischarge"
        elif "Q discharge/mA.h" == x_desire_key:
            x_name = "Qcharge"
        if y_desire_key == "time/s":
            y_name = "t"
        elif y_desire_key == "Ewe/V":
            y_name = "V"
        elif "x" == y_desire_key:
            y_name = "x"
        elif "<i>/mA" == y_desire_key:
            y_name = "i"
        elif "Q discharge/mA.h" == y_desire_key:
            y_name = "Qdischarge"
        elif "Q discharge/mA.h" == y_desire_key:
            y_name = "Qcharge"
        if rest_desire == "y":
            output_path = output_path / f"{file.stem}_{x_name}_vs_{y_name}_no-rest.txt"
        else:
            output_path = output_path / f"{file.stem}_{x_name}_vs_{y_name}.txt"
        np.savetxt(output_path, np.column_stack((x_values, y_values)))
        print(f"\tDone extracting data.\n\tPlease see the 'txt' directory for "
              f"two-column of file of the requested\n\tdata.\n\n\tPlotting "
              f"data...")
        xy_plotter(x_values, y_values, x_desire_key, y_desire_key, rest_desire, file.stem, plotpaths)
        plotfolders = [p.name for p in plotpaths]
        print(f"\tDone plotting data.\n\tPlease see the {plotfolders} folders.")

        return None


def xy_plotter(x_values, y_values, x_desire_key, y_desire_key, rest_desire, filename, plotpaths):
    if STYLE:
        plt.style.use(bg_mpl_style)
    fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
    if "time" in x_desire_key:
        if np.amax(x_values) > 100:
            x_values = x_values / 60**2
    if STYLE:
        ax.plot(x_values, y_values, lw=LINEWIDTH)
    else:
        ax.plot(x_values, y_values, color=COLOR, lw=LINEWIDTH)
    ax.set_xlim(np.amin(x_values), np.amax(x_values))
    ax.set_ylim(np.amin(y_values), np.amax(y_values))
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICKS)
    ax.minorticks_on()
    if x_desire_key == "time/s":
        xlabel = r"$t$ $[\mathrm{h}]$"
    elif x_desire_key == "Ewe/V":
        xlabel = r"$V$ $[\mathrm{V}]$"
    elif "x" == x_desire_key:
        xlabel = r"$x$"
    elif "<i>/mA" == x_desire_key:
        xlabel = r"$i$ $[\mathrm{mA}]$"
    elif "Q discharge/mA.h" == x_desire_key:
        xlabel = r"$Q_{\mathrm{discharge}}$ $[\mathrm{mAh}]$"
    elif "Q discharge/mA.h" == x_desire_key:
        xlabel = r"$Q_{\mathrm{charge}}$ $[\mathrm{mAh}]$"
    ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABELS)
    if y_desire_key == "time/s":
        ylabel = r"$t$ $[\mathrm{h}]$"
    elif y_desire_key == "Ewe/V":
        ylabel = r"$V$ $[\mathrm{V}]$"
    elif "x" == y_desire_key:
        ylabel = r"$x$"
    elif "<i>/mA" == y_desire_key:
        ylabel = r"$i$ $[\mathrm{mA}]$"
    elif "Q discharge/mA.h" == y_desire_key:
        ylabel = r"$Q_{\mathrm{discharge}}$ $[\mathrm{mAh}]$"
    elif "Q discharge/mA.h" == y_desire_key:
        ylabel = r"$Q_{\mathrm{charge}}$ $[\mathrm{mAh}]$"
    ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABELS)
    for p in plotpaths:
        if rest_desire == "y":
            plt.savefig(f"{p.name}/{filename}_no-rest.{p.name}", bbox_inches="tight")
        else:
            plt.savefig(f"{p.name}/{filename}.{p.name}", bbox_inches="tight")

    return None


def main():
    data_path = Path.cwd() / "data"
    if not data_path.exists():
        data_path.mkdir()
        print(f"{80*'-'}\nA folder called 'data' has been created.\
                \nPlease place your .txt echem files here.\
                \n{80*'-'}")
        sys.exit()
    txt_files = list(data_path.glob("*.txt"))
    if len(txt_files) == 0:
        print(f"{80*'-'}\nNo data files found.\
                \nPlease place your .txt echem files in the 'data' folder.\
                \n{80*'-'}")
        sys.exit()
    comma_to_dot_path = Path.cwd() / "comma_to_dot"
    if not comma_to_dot_path.exists():
        comma_to_dot_path.mkdir()
    comma_to_dot(txt_files, comma_to_dot_path)
    txt_path = Path.cwd() / "txt"
    if not txt_path.exists():
        txt_path.mkdir()
    txt_files_dotted = comma_to_dot_path.glob("*.txt")
    plotpaths = [Path.cwd() / "png", Path.cwd() / "pdf", Path.cwd() / "svg"]
    for p in plotpaths:
        if not p.exists():
            p.mkdir()
    print(f"{80*'-'}\nWorking with files...")
    ec_lab_extractor(txt_files_dotted, txt_path, plotpaths)
    print(f"{80*'-'}\nDone working with files.")

    return None


if __name__ == "__main__":
    main()

# End of file.
