import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
try:
    PLOT_STYLE = True
    from bg_mpl_stylesheet.bg_mpl_stylesheet import bg_mpl_style
except ImportError:
    PLOT_STYLE = None
from diffpy.utils.parsers.loaddata import loadData


DPI = 600
FIGSIZE = (12,4)
FONTSIZE_LABELS = 20
FONTSIZE_TICKS = 16
MAJOR_TICK_INDEX_TIME = 5
MAJOR_TICK_INDEX_VOLTAGE = 0.5
LINES_CHANGES = np.array([4803, 6162, 7632, 8500]) - 2

# TIME_CHANGES = np.array([4.910468837981852E+004,
#                          6.265857880506461E+004,
#                          7.732129662271368E+004,
#                          8.595690478000778E+004,
#                         ]) / 60**2

# index time[s] voltage [V] [x]
# 4803 4.910468837981852E+004	3.9000618E+000	1.147973829064084E-001
# 6162 6.265857880506461E+004	2.4999456E+000	3.647478028263231E-001
# 7632 7.732129662271368E+004	3.9000235E+000	9.425066130954751E-002
# 8500 8.595690478000778E+004	2.5000412E+000	2.530908301094141E-001	

def comma_to_to(file, output_path):
    with file.open(mode="r") as f:
        name = file.name
        s = f.read()
    with (output_path / name).open(mode="w") as o:
        o.write(s.replace(",", "."))

    return None


def csv_to_dict_extract(file):
    df = pd.read_csv(file, delimiter="\t")
    keys = [k for k in list(df.keys()) if not "Unnamed" in k]
    d = {}
    for k in keys:
        d[k] = df[k].to_numpy()

    return d


def dict_t_x_v_plot(d, output_paths):
    keys = list(d.keys())
    for k in keys:
        if "time/s" == k:
            t_key = k
        elif "Ewe/V" == k:
            v_key = k
        elif "x" == k:
            x_key = k
    t, v, x = d[t_key] / 60**2, d[v_key], d[x_key]
    for i in range(len(x)-1):
        if x[i] != x[i+1]:
            x_start_index = i + 1
            x_start = x[x_start_index]
            break
    if x[x_start_index + 1] - x[x_start_index] > 0:
        initial_mode = "discharge"
    elif x[x_start_index + 1] - x[x_start_index] < 0:
        initial_mode = "charge"
    complete = False
    x_change_indices = []
    rest_counter = 0
    if initial_mode == "charge":
        for i in range(x_start_index, len(x) - 1):
            if x[i+1] - x[i] > 0:
                x_change_index = i
                x_change_indices.append(i)
                mode = "discharge"
                break
            elif x[i+1] - x[i] == 0:
                rest_counter += 1
    elif initial_mode == "discharge":
        for i in range(x_start_index, len(x) - 1):
            if x[i+1] - x[i] < 0:
                x_change_index = i
                x_change_indices.append(i)
                mode = "charge"
                break
            elif x[i+1] - x[i] == 0:
                rest_counter += 1
    while complete is False:
        if mode == "discharge":
            for i in range(x_change_index + 1, len(x) - 1):
                if i == len(x) - 2:
                    complete = True
                elif x[i+1] - x[i] < 0:
                    x_change_index = i
                    x_change_indices.append(i)
                    mode = "charge"
                    break
        if mode == "charge":
            for i in range(x_change_index + 1, len(x) - 1):
                if i == len(x) - 2:
                    complete = True
                elif x[i+1] - x[i] > 0:
                    x_change_index = i
                    x_change_indices.append(i)
                    mode = "discharge"
                    break

    t_change = [t[e-rest_counter-2] for e in x_change_indices]
    v_change = [v[e-rest_counter-2] for e in x_change_indices]
    x_change = [x[e-rest_counter-2] for e in x_change_indices]

    # t_0 = t[x_start_index:x_change_indices[0]:]
    # v_0 = v[x_start_index:x_change_indices[0]:]
    # x_0 = x[x_start_index:x_change_indices[0]:]
    # t_1 = t[x_change_indices[0]:x_change_indices[1]:]
    # v_1 = v[x_change_indices[0]:x_change_indices[1]:]
    # x_1 = x[x_change_indices[0]:x_change_indices[1]:]
    # t_2 = t[x_change_indices[1]-1:x_change_indices[2]:]
    # v_2 = v[x_change_indices[1]-1:x_change_indices[2]:]
    # x_2 = x[x_change_indices[1]-1:x_change_indices[2]:]
    # t_3 = t[x_change_indices[2]-1:x_change_indices[3]:]
    # v_3 = v[x_change_indices[2]-1:x_change_indices[3]:]
    # x_3 = x[x_change_indices[2]-1:x_change_indices[3]:]

    xticks = np.arange(0, 1+0.1, 0.1)
    xticks = [float(e) for e in xticks]
    t0 = t[x_start_index]
    t, v, x = t[x_start_index::]-t0, v[x_start_index::], x[x_start_index::]
    t_change = t_change - t0
    t_range = np.amax(t) - np.amin(t)
    v_range = np.amax(v) - np.amin(v)
    xstep = abs(x[1] - x[0])
    d_xticks = {}
    for i in range(len(x)):
        # for j in range(len(t_change)):
        #     if np.isclose(t[i], t_change[j], atol=(t[1]-t[0]) / 2):
        #         d_xticks[i] = {"t": t[i], "x": f"{x[i]:.2f}", "v" : v[i]}
        #         print(i, t[i], v[i], x[i])
        #         break
            # continue
        for j in range(len(xticks)):
            if np.isclose(x[i], xticks[j], atol=xstep/2):
                d_xticks[i] = {"t": t[i], "x": f"{x[i]:.1f}", "v" : v[i]}
                continue
    t_xticks = [d_xticks[k]["t"] for k in d_xticks.keys()]
    t_xticks_labels = [d_xticks[k]["x"] for k in d_xticks.keys()]
    v_xticks = [d_xticks[k]["v"] for k in d_xticks.keys()]

    for i in range(len(t_xticks)-1):
        for j in range(len(t_change)):
            if t_xticks[i] < t_change[j] < t_xticks[i+1] and t_change[j] not in t_xticks:
                t_xticks.insert(i, t_change[j])
                v_xticks.insert(i, v_change[j])
                t_xticks_labels.insert(i, f"{x_change[j]:.2f}")
                break
    t_xticks.append(t_change[-1])
    v_xticks.append(v_change[-1])
    t_xticks_labels.append(f"{x_change[-1]:.2f}")
    remove_indices = [i for i in range(len(t_xticks_labels)) if t_xticks_labels[i] == "0.1"]
    counter = 0
    for i in range(len(remove_indices)):
        del t_xticks[remove_indices[i]]
        del v_xticks[remove_indices[i]]
        del t_xticks_labels[remove_indices[i]]
        counter += 1
    # t = t - t0
    # t_xticks = np.array(t_xticks) - t0
    # for k in d_xticks.keys():
    #     print(k, d_xticks[k]["t"], d_xticks[k]["x"])
    # sys.exit()
    # print(t_xticks)
    # print(t_xticks_labels)
    # sys.exit()

    if not isinstance(PLOT_STYLE, type(None)):
        plt.style.use(bg_mpl_style)
    fig = plt.figure(dpi=DPI, figsize=FIGSIZE)
    ax0 = fig.add_subplot(111)
    ax1 = ax0.twiny()
    ax1.plot(t, v)
    ax1.set_xlim(np.amin(t), np.amax(t))
    ax1.set_ylim(np.amin(v), np.amax(v))
    ax0.set_xlim(np.amin(t), np.amax(t))
    ax0.set_ylim(np.amin(v), np.amax(v))
    ax0.set_xticks(t_xticks)
    ax0.set_xticklabels(t_xticks_labels)
    ax0.set_xlabel("$x$ in Li$_{x}$", fontsize=FONTSIZE_LABELS)
    ax1.set_xlabel("$t$ $[\mathrm{h}]$", fontsize=FONTSIZE_LABELS)
    ax0.set_ylabel("$E_{\mathrm{we}}$ vs. Li/Li$^{+}$", fontsize=FONTSIZE_LABELS)
    ax1.xaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_TIME))
    ax1.xaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_TIME / 5))
    ax1.yaxis.set_major_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE))
    ax1.yaxis.set_minor_locator(MultipleLocator(MAJOR_TICK_INDEX_VOLTAGE / 5))
    ax0.xaxis.set_tick_params(labelsize=FONTSIZE_TICKS)
    ax1.xaxis.set_tick_params(labelsize=FONTSIZE_TICKS)
    ax0.yaxis.set_tick_params(labelsize=FONTSIZE_TICKS)
    # ax0.xaxis.set_tick_params(labelsize=FONTSIZE_TICKS)
    # ax0.scatter(np.array(t_change), np.ones(len(t_change))*np.amin(v), marker="x")
    # ax0.annotate("/", (t_change[0], np.min(v)-0.1))
    # pos = ax1.get_position()
    # corners = [pos.x0, pos.y0, pos.width, pos.height]
    for i in [0, 1, 3]:
        plt.text(t_change[i] - 0.0175 * t_range, np.amin(v) - 0.02*v_range, "|", rotation=45)
    plt.text(t_change[2] - 0.014 * t_range, np.amin(v) - 0.02*v_range, "|", rotation=45)
    for p in output_paths:
        plt.savefig(f"{p}/biologic_x_v_plot.{p}", bbox_inches="tight")
    plt.close()
    # plt.show()
    sys.exit()


    # for i in range(len(xticks_indices)):
    #     print(x[xticks_indices[i]], t[xticks_indices[i]])
    #             t_x_ticks[0]
    #         print(i, x[i])
    # print(t_xticks)
    # print(xticks)
    # sys.exit()
    # print(round(np.log10(abs(x_0[2]-x_0[1])), 0))
    # print(np.log10(abs(x_0[2]-x_0[1])))
    # print(x_0[2]-x_0[1])
    # sys.exit()
    # x_list, v_list = [x_0, x_1, x_2, x_3], [v_0, v_1, v_2, v_3]
    # vmin, vmax = np.amin(v[x_start_index::]), np.amax(v[x_start_index::])
    # xmins, xmaxs = [np.amin(e) for e in x_list], [np.amax(e) for e in x_list]
    # xranges = [xmaxs[i] - xmins[i] for i in range(len(xmins))]
    #
    # x_ticks = [x_ticks_0, x_ticks_1, x_ticks_2]
    # fig, axs = plt.subplots(dpi=DPI,
    #                         figsize=FIGSIZE,
    #                         nrows=1,
    #                         ncols=4,
    #                         sharey=True,
    #                         )
    # fig.subplots_adjust(wspace=0.0)
    # for i in range(0, len(x_change) - 1):
    #     axs[i].spines["right"].set_visible(False)
    # for i in range(1, len(x_change)):
    #     axs[i].spines["left"].set_visible(False)
    #     axs[i].get_yaxis().set_visible(False)
    # for i in range(len(x_change)):
    #     axs[i].set_ylim(vmin, vmax)
    # offset = 0.001
    # for i in range(len(x_list)):
    #     axs[i].plot(x_list[i], v_list[i])
    #     axs[i].set_xlim(np.amin(x_list[i]) - offset * xranges[i],
    #                     np.amax(x_list[i]) + offset * xranges[i]
    #                     )
    #     axs[i].set_ylim(vmin, vmax)
    # for i in range(len(x_ticks)):
    #     axs[i].set_xticks(x_ticks[i])
    # for i in [0, 2]:
    #     axs[i].invert_xaxis()
    # d = 1 - (FIGSIZE[0] / FIGSIZE[1])**-1  # proportion of vertical to horizontal extent of the slanted line
    # kwargs = dict(marker=[(-1, -d), (1, d)],
    #               markersize=12,
    #               linestyle="none",
    #               color='k',
    #               mec='k',
    #               mew=1,
    #               clip_on=False,
    #               )
    # for i in range(1, len(x_change)):
    #     axs[i].plot([0, 1], [0, 0], transform=axs[i].transAxes, **kwargs)
    # # axs[1].plot([0, 0], transform=axs[1].transAxes, **kwargs)
    # for p in output_paths:
    #     plt.savefig(f"{p}/biologic_x_v_plot.{p}", bbox_inches="tight")
    # plt.show()

    return

def main():
    data_path = Path.cwd() / "data"
    if not data_path.exists():
        data_path.mkdir()
        print(f"{80*'-'}\nA folder called 'data' has been created.\nPlease "
              f"place your .txt files here and rerun the program.\n{80*'-'}")
        sys.exit()
    data_files = list(data_path.glob("*.txt"))
    if len(data_files) == 0:
        print(f"{80*'-'}\nNo .txt files were found in the 'data' folder.\n"
              f"Please place your .txt there and rerun the program.\n{80*'-'}")
        sys.exit()
    data_comma_to_dot_path = Path.cwd() / "data_comma_to_dot"
    if not data_comma_to_dot_path.exists():
        data_comma_to_dot_path.mkdir()
    print(f"{80*'-'}\nReplacing commas with dots...")
    for file in data_files:
        print(f"\t{file.name}")
        comma_to_to(file, data_comma_to_dot_path)
    data_files = list(data_comma_to_dot_path.glob("*.txt"))
    print(f"Done replacing commas with dots.\n{80*'-'}\nExtracting data...")
    data_dict = {}
    for file in data_files:
        print(f"\t{file.name}")
        data_dict[file.name] = csv_to_dict_extract(file)
    print(f"Done extracting data.\n{80*'-'}\nPlotting data...")
    output_folders = ["pdf", "png", "svg"]
    for folder in output_folders:
        if not (Path.cwd() / folder).exists():
            (Path.cwd() / folder).mkdir()
    for k in list(data_dict.keys()):
        dict_t_x_v_plot(data_dict[k], output_folders)



    return None


if __name__ == "__main__":
    main()

# End of file.
