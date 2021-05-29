%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams["animation.html"] = "jshtml"
import matplotlib.animation as animation
import numpy as np
import nhpp
from sim_utils import animframe, set_ax_lim
from nodespace import NodeSpace
import time
from matplotlib.collections import PatchCollection


# initialize system
ns = NodeSpace()
n_clusts = 3
n_points = [7, 7, 7]
centers = [[-6,0], [0,6], [8,-6]]
radius = [1.5,1.5,1.5]
v = [[1,0], [0,-1], [-1,1]]
a =  [[0,-0.1], [0.1,0], [0,-0.1]]
z0 = ns.init_clusters(n_clusts, n_points, centers, radius)
v0, a0 = ns.init_dynamics(n_clusts, n_points, v, a)
ns.init_conditions(z0, v0, a0)


selected_nodes = [(0,7), (8, 20), (1, 9), (3,19)]

# animation

fig, axtop, axes = animframe(2,2,np.linspace(0,16), selected_nodes)

points, = axtop.plot([], [], "o", ms=6)


lines = []
for ax in axes:
    line, = ax.plot([], [], lw="2")
    lines.append(line)


def init():
    points.set_data([], [])
    for line in lines:
        line.set_data([],[])
    all_lines = [points] + lines
    return all_lines



t = np.linspace(0, 15)
y = np.sin(t)

# %%
intensities = []
for (u,v) in selected_nodes:
    intensity = []
    for t_i in t:
        intensity.append(ns.lambda_fun(t_i, u, v))
    
    intensity = np.array(intensity)
    intensities.append(intensity)
    
intensities = np.array(intensities)

#%%
rmat = nhpp.root_matrix(ns) 
mmat = nhpp.monotonicity_mat(ns, rmat)

print("Simulating events...")
t0 = time.time()
nhppmat = nhpp.nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
print("Elapsed simulation time (s): ", time.time() - t0)

set_ax_lim(axtop, xlim=[-10,10], ylim=[-10,10])
for i, ax in enumerate(axes):
    ymax = np.max(intensities[i])
    set_ax_lim(ax, xlim=[t[0], t[-1]], ylim=[0,ymax])
    
    # draw event times
    cur_nodes = selected_nodes[i]
    cur_events = nhpp.get_entry(nhppmat, cur_nodes[0], cur_nodes[1])
    ax.vlines(x=cur_events, ymin=0, ymax=ymax, color="green", linestyles="solid", alpha=0.1)


# %%
def update(i, t, intensities, points, lines, nodes):

    # points in latent space
    z = ns.step(t[i])
    points.set_data(z[:,0],z[:,1])

    # intensity rates with event times for selected nodes
    for idx, line  in enumerate(lines):
        line.set_data(t[:i], intensities[idx][:i])

    all_lines = [points] + lines

    return all_lines

anim = animation.FuncAnimation(fig, update, fargs=[t, intensities, points, lines, selected_nodes], init_func=init,frames=50,interval=200, blit=True)
anim.save('animation_frame.mp4', dpi=500, fps=30, extra_args=['-vcodec', 'libx264'])
anim

# %%


# %%
