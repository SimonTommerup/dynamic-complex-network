import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import webcolors
import numpy as np

def ax_idx(r, c, n_cols):
    idx = r * n_cols + c
    return idx 

def sharex_grid(rows, cols, axes):
    for c in range(cols):
        ax0 = axes[ax_idx(0, c, n_cols=cols)]
        for r in range(1, rows):
            ax = axes[ax_idx(r, c, n_cols=cols)]
            ax0.get_shared_x_axes().join(ax0, ax)

            if r < rows-1:
                ax.set_xticklabels([])
        ax0.set_xticklabels([])

def animframe(rows, cols, selected_nodes):
    fig = plt.figure(constrained_layout=False)

    outer_grid = gridspec.GridSpec(2,1, height_ratios=[2,1], hspace=0.5)
    upper_cell = outer_grid[0,0]
    lower_cell = outer_grid[1,0]

    axtop = fig.add_subplot(upper_cell)
    axtop.set_title("Latent space")

    lower_cell_grid = gridspec.GridSpecFromSubplotSpec(rows, cols, lower_cell, hspace=1.25)
    axes = []

    for i in range(rows):
        for j in range(cols):
            ax = fig.add_subplot(lower_cell_grid[i, j])
            axes.append(ax)

    for i, ax in enumerate(axes):
        cur_nodes = r"{}".format(selected_nodes[i]).strip("()")
        ax.set_title(r"$\lambda_{{{:2s}}}$".format(cur_nodes))

    return fig, axtop, axes

def set_ax_lim(ax, xlim, ylim):
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

def rgb_arr(name):
    rgb = webcolors.name_to_rgb(name)
    tup = rgb_to_tup(rgb)
    arr = np.array(tup)
    return arr

def rgb_to_tup(rgb):
    tup = [0,0,0]
    for idx, val in enumerate(rgb):
        tup[idx] = val / 255.
    return tuple(tup)