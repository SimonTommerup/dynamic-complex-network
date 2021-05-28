%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams["animation.html"] = "jshtml"
import matplotlib.animation as animation
import numpy as np


fig = plt.figure(constrained_layout=False)

outer_grid = gridspec.GridSpec(2,1, height_ratios=[1,2], hspace=0.5)
upper_cell = outer_grid[0,0]
lower_cell = outer_grid[1,0]

axtop = fig.add_subplot(upper_cell)
axtop.set_title("Upper")
axtop.set_xlim(0,np.pi)
axtop.set_ylim(-1,1)

rows, cols = 2, 2
lower_cell_grid = gridspec.GridSpecFromSubplotSpec(rows, cols, lower_cell, hspace=1.25)

axes = []
for i in range(rows):
    for j in range(cols):
        ax = fig.add_subplot(lower_cell_grid[i, j])
        ax.set_xlim(0,np.pi)
        ax.set_ylim(-1,1)
        ax.set_title(f"Plot {i},{j}")
        axes.append(ax)

anno_opts = dict(xy=(0.5, 0.5), xycoords='axes fraction',
                 va='center', ha='center', color="red")

def ax_idx(r, c, n_cols):
    idx = r * n_cols + c
    return idx 

for r in range(rows):
    for c in range(cols):
        ax = axes[ax_idx(r, c, n_cols=cols)]
        ax.annotate(f"{r,c}", **anno_opts)

def sharex_grid(rows, cols, axes):
    for c in range(cols):
        ax0 = axes[ax_idx(0, c, n_cols=cols)]
        for r in range(1, rows):
            ax = axes[ax_idx(r, c, n_cols=cols)]
            ax0.get_shared_x_axes().join(ax0, ax)

            if r < rows-1:
                ax.set_xticklabels([])
        ax0.set_xticklabels([])
    
sharex_grid(rows, cols, axes)

top_line, = axtop.plot([],[])
lines = []
for ax in axes:
    line, = ax.plot([], [])
    #line, = ax.plot([0,np.pi],[-1,1])
    lines.append(line)


def init():
    top_line.set_data([], [])
    for line in lines:
        line.set_data([],[])
    all_lines = [top_line] + lines
    return all_lines


x = np.linspace(0, np.pi, 20)
y = np.sin(x)

# animation function.  This is called sequentially
def update(i, x, y, top_line, lines):
    top_line.set_data(x[:i],y[:i])
    for line in lines:
        line.set_data(x[:i], y[:i])

    all_lines = [top_line] + lines

    return all_lines

anim = animation.FuncAnimation(fig, update, fargs=[x, y, top_line, lines], init_func=init,frames=20,interval=200, blit=True)
anim.save('animation_frame.mp4', dpi=500, fps=30, extra_args=['-vcodec', 'libx264'])
anim

#plt.tight_layout() # incompatible with hspace
#plt.show()


