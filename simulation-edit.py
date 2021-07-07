%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams["animation.html"] = "jshtml"
import matplotlib.animation as animation
import numpy as np
import nhpp
import torch
from sim_utils import animframe, set_ax_lim, rgb_arr
from nodespace import NodeSpace
import time
from matplotlib.collections import PatchCollection
from sklearn.datasets import make_blobs

#plt.style.use("dark_background")

# initialize system
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

ZERO_SEED = 1
np.random.seed(ZERO_SEED)
torch.manual_seed(ZERO_SEED)
np.seterr(all='raise')

LOAD_PATH = "50point-newclust-beta=5" + ".npy"

ns =NodeSpace()
ns.beta = 5
n_points = 50

centers = np.array([[-4.,0.0],[4.,0.],[4,-4.]])
z_gt, y = make_blobs(n_samples=n_points, centers=centers, center_box=(-10,10))
vdir = np.array([[0.04, -0.04], [-0.02, -0.04], [-0.04, 0.02] ])
adir = np.array([[-0.001, -0.001], [0.001, -0.001], [0.001, 0.001] ])
v_gt, a_gt = ns.custom_init_dynamics(n_points, y, vdir, adir)
ns.init_conditions(z_gt, v_gt, a_gt)


t = np.linspace(0, 15)

#%%
selected_node_tups = [(7,10), (3, 49), (14, 27), (33,5)]
#selected_node_tups = [(0,7), (1,9)]
selected_node_list = [n for tup in selected_node_tups for n in tup]
selected_node_ind = [True if val in selected_node_list else False for val in range(n_points)]
other_node_ind = [not val for val in selected_node_ind]

fig, axtop, axes = animframe(2,2, selected_node_tups)
#fig, axtop, axes = animframe(rows=1,cols=2, selected_nodes=selected_node_tups)
colors = ["indianred", "tan", "mediumseagreen", "royalblue"]
#colors = ["indianred", "royalblue"]
selected_points = []
for idx, tup in enumerate(selected_node_tups):
    selected_point = axtop.scatter([],[])
    selected_points.append(selected_point)

other_points = axtop.scatter([],[])

legend = axtop.legend()

link, = axtop.plot([],[], linewidth=1)

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

    link.set_data([],[])

    for line in lines:
        line.set_data([],[])

    all_lines = selected_points + [other_points] + [link] + lines + [legend] # [ snl ] ? 

    return all_lines


intensities = []
for (u,v) in selected_node_tups:
    intensity = []
    for t_i in t:
        intensity.append(ns.lambda_fun(t_i, u, v))
    
    intensity = np.array(intensity)
    intensities.append(intensity)
    
intensities = np.array(intensities)

#rmat = nhpp.root_matrix(ns) 
#mmat = nhpp.monotonicity_mat(ns, rmat)
# print("Simulating events...")
# t0 = time.time()
# nhppmat = nhpp.nhpp_mat(ns=ns, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
# print("Elapsed simulation time (s): ", time.time() - t0)

print("Loading events:")
nhppmat = np.load(LOAD_PATH, allow_pickle=True)

set_ax_lim(axtop, xlim=[-15,15], ylim=[-15,15])
all_events = []
nodes_to_events = []
for i, ax in enumerate(axes):
    ymax = np.max(intensities[i])
    set_ax_lim(ax, xlim=[t[0], t[-1]], ylim=[0,ymax])
    
    # draw event times
    cur_nodes = selected_node_tups[i]
    cur_events = nhpp.get_entry(nhppmat, cur_nodes[0], cur_nodes[1])
    all_events.append(cur_events)
    nodes_to_events.append([cur_nodes, cur_events])
    ax.vlines(x=cur_events, ymin=0, ymax=ymax, color="green", linestyles="solid", alpha=0.1)

all_events = np.concatenate(all_events, axis=0)
all_events = np.sort(all_events, axis=0)
#%%

#%%
event_time_to_node_pair = {}
for node_pair, event_times in nodes_to_events:
    for event_time in event_times:
        event_time_to_node_pair[event_time] = node_pair


intensities_at_event = []
for (u,v) in selected_node_tups:
    intensity = []
    for t_i in all_events:
        intensity.append(ns.lambda_fun(t_i, u, v))
    
    intensity = np.array(intensity)
    intensities_at_event.append(intensity)
    
intensities_at_event = np.array(intensities_at_event)

# %%
def update(i, t, intensities, selected_points, other_points, link, lines, legend):
    z = ns.step(t[i])
    for idx, selected_point in enumerate(selected_points):
        cur_node_pair = selected_node_tups[idx]
        cur_slice = [True if n in cur_node_pair else False for n in range(n_points)]
        selected_point.set_offsets(ns.z[cur_slice, :])
        cur_col = rgb_arr(colors[idx])
        selected_point.set_color([cur_col, cur_col])
        selected_point.set_alpha(1)
        selected_point.set_label(f"({cur_node_pair[0]},{cur_node_pair[1]})")
    
    legend.remove()
    legend = axtop.legend(loc="lower left", ncol=len(selected_node_tups),prop={"size":6})

    other_points.set_offsets(z[other_node_ind,:])
    other_points.set_color(rgb_arr("black"))
    other_points.set_alpha(0.1)
    linked_nodes = event_time_to_node_pair[t[i]]
    node_u = linked_nodes[0]
    node_v = linked_nodes[1]
    xdata= [ns.z[node_u,0], ns.z[node_v,0]]
    ydata = [ns.z[node_u,1], ns.z[node_v,1]]
    link.set_data(xdata,ydata)
    link.set_color("black")

    for idx, line  in enumerate(lines):
        line.set_data(t[:i], intensities[idx][:i])

    all_lines = selected_points  + [other_points] + [link] + lines + [legend] 
    return all_lines



print("Animating...")
t1 = time.time()
#anim = animation.FuncAnimation(fig, update, fargs=[t, intensities, selected_points, other_points, lines, legend], init_func=init,frames=len(t),interval=1e3, blit=True)
anim = animation.FuncAnimation(fig, update, fargs=[all_events, intensities_at_event, selected_points, other_points, link, lines, legend], init_func=init,frames=len(all_events),interval=1e3, blit=True)
anim.save('50point-net.mp4', dpi=500, fps=10, extra_args=['-vcodec', 'libx264'])
anim
#print("Elapsed animation time (s): ", time.time() - t1)


#%%

anim
# %%
