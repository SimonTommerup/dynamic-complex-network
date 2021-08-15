%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams["animation.html"] = "jshtml"
import matplotlib.animation as animation
import numpy as np
import nhpp
from sim_utils import animframe, set_ax_lim, rgb_arr
from nodespace import NodeSpace
import time
from matplotlib.collections import PatchCollection

#plt.style.use("dark_background")

# # initialize system
# ns = NodeSpace()
# n_clusts = 3
# points_in_clusts = [7, 7, 7]
# n_points = sum(points_in_clusts)
# centers = [[-6,0], [0,6], [8,-6]]
# radius = [1.5,1.5,1.5]
# v = [[1,0], [0,-1], [-1,1]]
# a =  [[0,-0.1], [0.1,0], [0,-0.1]]
# z0 = ns.init_clusters(n_clusts, points_in_clusts, centers, radius)
# v0, a0 = ns.init_dynamics(n_clusts, points_in_clusts, v, a)
# ns.init_conditions(z0, v0, a0)
ns = NodeSpace()
z_gt = np.array([[-3, 1.], [-3., -1.], [3.,1.], [3.,-1.]])
v_gt = np.array([[2.0,-2.0], [2.0,2.0], [-2.0,-2.0], [-2.0, 2.0]])
a_gt = np.array([[-0.5,0.5], [-0.5,-0.5], [0.5,0.5], [0.5, -0.5]])
n_points=len(z_gt)
ns.init_conditions(z_gt, v_gt, a_gt)

# beta and alpha: lambda = exp(beta - alpha * dist)
ns.beta = 5

t = np.linspace(0, 15)

#%%
#selected_node_tups = [(0,7), (8, 20), (1, 9), (3,19)]
selected_node_tups = [(0,1), (2,3)]
selected_node_list = [n for tup in selected_node_tups for n in tup]
selected_node_ind = [True if val in selected_node_list else False for val in range(n_points)]
other_node_ind = [not val for val in selected_node_ind]

#fig, axtop, axes = animframe(2,2,np.linspace(0,16), selected_node_tups)
fig, axtop, axes = animframe(1,2, selected_node_tups)
#colors = ["indianred", "tan", "mediumseagreen", "royalblue"]
colors = ["indianred", "tan"]
#colors = ["indianred"]
selected_points = []
for idx, tup in enumerate(selected_node_tups):
    selected_point = axtop.scatter([],[])
    selected_points.append(selected_point)

other_points = axtop.scatter([],[])

legend = axtop.legend()

lines = []
for ax in axes:
    line, = ax.plot([], [], lw="1")
    lines.append(line)

def init():
    for idx, selected_point in enumerate(selected_points):
        cur_node_pair = selected_node_tups[idx]
        cur_slice = [True if n in cur_node_pair else False for n in range(n_points)]
        selected_point.set_offsets(ns.z[cur_slice, :])
        cur_col = rgb_arr(colors[idx])
        selected_point.set_color([cur_col, cur_col])
        selected_point.set_alpha(0.75)
        selected_point.set_label(f"({cur_node_pair[0]},{cur_node_pair[1]})")

    legend = axtop.legend(loc="lower left", ncol=len(selected_node_tups), prop={"size":6})

    other_points.set_offsets(ns.z0[other_node_ind,:])
    other_points.set_color(rgb_arr("black"))
    other_points.set_alpha(0.25)

    for line in lines:
        line.set_data([],[])

    all_lines = selected_points + [other_points] + lines + [legend] # [ snl ] ? 

    return all_lines


intensities = []
for (u,v) in selected_node_tups:
    intensity = []
    for t_i in t:
        intensity.append(ns.lambda_fun(t_i, u, v))
    
    intensity = np.array(intensity)
    intensities.append(intensity)
    
intensities = np.array(intensities)

rmat = nhpp.root_matrix(ns) 
mmat = nhpp.monotonicity_mat(ns, rmat)

print("Simulating events...")
t0 = time.time()
nhppmat = nhpp.nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
print("Elapsed simulation time (s): ", time.time() - t0)

set_ax_lim(axtop, xlim=[-15,15], ylim=[-15,15])
for i, ax in enumerate(axes):
    ymax = np.max(intensities[i])
    set_ax_lim(ax, xlim=[t[0], t[-1]], ylim=[0,ymax])
    
    # draw event times
    cur_nodes = selected_node_tups[i]
    cur_events = nhpp.get_entry(nhppmat, cur_nodes[0], cur_nodes[1])
    ax.vlines(x=cur_events, ymin=0, ymax=ymax, color="green", linestyles="solid", alpha=0.1)


# %%
def update(i, t, intensities, selected_points, other_points, lines, legend):
    z = ns.step(t[i])
    for idx, selected_point in enumerate(selected_points):
        cur_node_pair = selected_node_tups[idx]
        cur_slice = [True if n in cur_node_pair else False for n in range(n_points)]
        selected_point.set_offsets(ns.z[cur_slice, :])
        cur_col = rgb_arr(colors[idx])
        selected_point.set_color([cur_col, cur_col])
        selected_point.set_alpha(0.75)
        selected_point.set_label(f"({cur_node_pair[0]},{cur_node_pair[1]})")
    
    legend.remove()
    legend = axtop.legend(loc="lower left", ncol=len(selected_node_tups),prop={"size":6})

    other_points.set_offsets(z[other_node_ind,:])
    other_points.set_color(rgb_arr("black"))
    other_points.set_alpha(0.25)
    
    for idx, line  in enumerate(lines):
        line.set_data(t[:i], intensities[idx][:i])

    all_lines = selected_points  + [other_points] + lines + [legend] 
    return all_lines

print("Animating...")
t1 = time.time()
anim = animation.FuncAnimation(fig, update, fargs=[t, intensities, selected_points, other_points, lines, legend], init_func=init,frames=len(t),interval=1e3, blit=True)
anim.save('roots-report.mp4', dpi=500, fps=10, extra_args=['-vcodec', 'libx264'])
anim
#print("Elapsed animation time (s): ", time.time() - t1)
