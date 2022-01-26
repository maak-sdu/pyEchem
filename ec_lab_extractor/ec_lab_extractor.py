import sys
from pathlib import Path
import numpy as np
import pandas as pd


def comma_to_dot(txt_files):
    for file in txt_files:
        with file.open() as f:
            txt = f.read().replace(",", ".")
        output_path_file = Path.cwd() / "ec_lab_time_voltage_extracted" / file.name
        with open(output_path_file, mode="w") as o:
            o.write(txt)

    return output_path_file


def ec_lab_extracter(txt_dotted_path):
    with txt_dotted_path.open() as f:
        df = pd.read_csv(f, delimiter="\t")
    x, current = None, None
    for k in df.keys():
        if "time" in k:
            time = df[k].to_numpy()
        elif "Ewe" in k:
            voltage = df[k].to_numpy()
        elif k == "x":
            x = df[k].to_numpy()
        elif "<i>/mA" in k:
            i = df[k].to_numpy()
    if not isinstance(x, type(None)):
        for i in range(1, len(x)):
            if x[i-1] == x[0] and x[i] != x[i-1]:
                x_start_index = i
        time, voltage = time[x_start_index:], voltage[x_start_index:]
    elif not isinstance(current, type(None)):
        for i in range(1, len(current)):
            if current[i-1] == current[0] and current[i] != current[i-1]:
                current_start_index = i
        time, voltage = time[current_start_index:], voltage[current_start_index:]
    time = time - time[0]
    output_path = Path.cwd() / "ec_lab_time_voltage_extracted" / txt_dotted_path.name
    # header_extracted = f"{time_key}\t{voltage_key}"
    np.savetxt(output_path, np.column_stack((time, voltage)),
               # header=header_extracted,
               )

    return


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
    output_path = Path.cwd() / "ec_lab_time_voltage_extracted"
    if not output_path.exists():
        output_path.mkdir()
    txt_dotted_path = comma_to_dot(txt_files)
    ec_lab_extracter(txt_dotted_path)

    return None

if __name__ == "__main__":
    main()

# End of file.
