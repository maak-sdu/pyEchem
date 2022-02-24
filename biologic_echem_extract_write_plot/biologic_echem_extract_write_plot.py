import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DPI=300
FIGSIZE = (8,4)
FONTSIZE_LABELS = 20
FONTSIZE_TICKS = 14
LINEWIDTH = 1
COLORS = dict(bg_blue='#0B3C5D', bg_red='#B82601', bg_green='#1c6b0a',
              bg_lightblue='#328CC1', bg_grey='#a8b6c1', bg_yellow='#D9B310',
              bg_palered='#984B43', bg_maroon='#76323F', bg_palegreen='#626E60',
              bg_palebrown='#AB987A', bg_paleyellow='#C09F80')
COLOR = COLORS['bg_blue']


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
          f"'data/comma_to_dot' folder.")

    return None


def ec_lab_extractor(txt_files_dotted, output_path):
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
        xy_plotter(x_values, y_values, x_desire_key, y_desire_key, rest_desire, file.stem)

        return None


def xy_plotter(x_values, y_values, x_desire_key, y_desire_key, rest_desire, filename):
    fig, ax = plt.subplots(dpi=DPI, figsize=FIGSIZE)
    if "time" in x_desire_key:
        if np.amax(x_values) > 100:
            x_values = x_values / 60**2
    plt.plot(x_values, y_values, color=COLOR, lw=LINEWIDTH)
    plt.xlim(np.amin(x_values), np.amax(x_values))
    plt.ylim(np.amin(y_values), np.amax(y_values))
    plt.xticks(fontsize=FONTSIZE_TICKS)
    plt.yticks(fontsize=FONTSIZE_TICKS)
    if x_desire_key == "time/s":
        plt.xlabel(r"$t$ $[\mathrm{h}]$", fontsize=FONTSIZE_LABELS)
    elif x_desire_key == "Ewe/V":
        plt.xlabel(r"$V$ $[\mathrm{V}]$", fontsize=FONTSIZE_LABELS)
    elif "x" == x_desire_key:
        plt.xlabel(r"$x$", fontsize=FONTSIZE_LABELS)
    elif "<i>/mA" == x_desire_key:
        plt.xlabel(r"$i$ $[\mathrm{mA}]$", fontsize=FONTSIZE_LABELS)
    elif "Q discharge/mA.h" == x_desire_key:
        plt.xlabel(r"$Q_{\mathrm{discharge}}$ $[\mathrm{mAh}]$", fontsize=FONTSIZE_LABELS)
    elif "Q discharge/mA.h" == x_desire_key:
        plt.xlabel(r"$Q_{\mathrm{charge}}$ $[\mathrm{mAh}]$", fontsize=FONTSIZE_LABELS)
    if y_desire_key == "time/s":
        plt.ylabel(r"$t$ $[\mathrm{h}]$", fontsize=FONTSIZE_LABELS)
    elif y_desire_key == "Ewe/V":
        plt.ylabel(r"$V$ $[\mathrm{V}]$", fontsize=FONTSIZE_LABELS)
    elif "x" == y_desire_key:
        plt.ylabel(r"$x$", fontsize=FONTSIZE_LABELS)
    elif "<i>/mA" == y_desire_key:
        plt.ylabel(r"$i$ $[\mathrm{mA}]$", fontsize=FONTSIZE_LABELS)
    elif "Q discharge/mA.h" == y_desire_key:
        plt.ylabel(r"$Q_{\mathrm{discharge}}$ $[\mathrm{mAh}]$", fontsize=FONTSIZE_LABELS)
    elif "Q discharge/mA.h" == y_desire_key:
        plt.ylabel(r"$Q_{\mathrm{charge}}$ $[\mathrm{mAh}]$", fontsize=FONTSIZE_LABELS)
    plotfolders = ["png", "pdf"]
    for folder in plotfolders:
        if not (Path.cwd() / folder).exists():
            (Path.cwd() / folder).mkdir()
    if rest_desire == "y":
        plt.savefig(f"png/{filename}_no-rest.png", bbox_inches="tight")
        plt.savefig(f"png/{filename}_no-rest.pdf", bbox_inches="tight")
    else:
        plt.savefig(f"png/{filename}.png", bbox_inches="tight")
        plt.savefig(f"pdf/{filename}.pdf", bbox_inches="tight")

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
    print("Working with files...")
    ec_lab_extractor(txt_files_dotted, txt_path)
    print(f"{80*'-'}\nDone working with files.\n{80*'-'}\nPlease see the 'txt' "
          f"directory for two-column of files of the requested data.\nPlease "
          f"see the 'pdf' and 'png' folders for plots.\n{80*'-'}")

    return None


if __name__ == "__main__":
    main()

# End of file.
