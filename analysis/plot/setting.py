import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex

COLOR_CYCLE = "okabe-ito"


# avoid Type-3 fonts in PDF and PS files
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# set font family to STIX
mpl.rcParams["font.family"] = "STIXGeneral"
mpl.rcParams["mathtext.fontset"] = "stix"
# set font sizes
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 9
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9

# color palettes
num_colors_from_seq = 8
color_palettes = {
    "ieee": [
        "#0073B3",
        "#DE8300",
        "#009F74",
        "#D55E00",
        "#56B4E9",
        "#CC79A7",
        "#F0E442",
        "#000000",
    ],
    "okabe-ito": [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
        "#000000",
    ],
    "set2": [to_hex(c) for c in mpl.colormaps.get_cmap("Set2").colors],
    "set3": [to_hex(c) for c in mpl.colormaps.get_cmap("Set3").colors],
    "viridis": [
        to_hex(c)
        for c in mpl.colormaps.get_cmap("viridis")(
            np.linspace(0.1, 0.9, num_colors_from_seq)
        )
    ],
    "plasma": [
        to_hex(c)
        for c in mpl.colormaps.get_cmap("plasma")(
            np.linspace(0.1, 0.9, num_colors_from_seq)
        )
    ],
}
# line styles:
# '-': solid, '--': dashed, ':': dotted, '-.': dashdot
linestyle_list = ["-", "--", ":", "-."]
# marker styles:
# 'o': circle, 's': square, 'v': triangle_down, '^': triangle_up
# '<': triangle_left, '>': triangle_right, 'D': diamond, 'p': pentagon
# 'h': hexagon, 'x': cross, 'X': filled cross, '+': plus, '*': star
marker_list = ["o", "s", "v", "^", "D"]

# change the default full cycle on demand
full_cycler = (
    cycler(color=color_palettes[COLOR_CYCLE])
    # * cycler(linestyle=linestyle_list)
    # * cycler(marker=marker_list)
)

# global setting
plt.rcParams["axes.prop_cycle"] = cycler(color=color_palettes[COLOR_CYCLE])

# use the following config to reset color cycle in each axes
# ax.set_prop_cycle(cycler(color=palettes[color_cycle]))
