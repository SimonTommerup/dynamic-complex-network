#%%
%matplotlib inline
import numpy as np
import torch
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from IPython.display import HTML
import matplotlib.animation as animation

plt.style.use("seaborn")

class NodeSpace():
    def __init__(self):
        self.z0 = None
        self.v0 = None
        self.a0 = None
        self.z = None
        self.v = None
        self.a = None
    
    def step(self, dt):
            self.z = self.z0[:,:] + self.v0[:,:]*dt + 0.5*self.a0[:,:]*dt**2
            return self.z

    def init_conditions(self, z0, v0, a0):
        self.z0 = z0
        self.v0 = v0
        self.a0 = a0
        self.z = z0
        self.v = v0
        self.a = a0

    def init_clusters(self, n_clusts, n_points, centers, rads, seed=0):
        np.random.seed(seed)
        clusts = []
        for c in range(n_clusts):
            clust = self.init_points(n_points[c], centers[c], rads[c])
            clusts.append(clust)
        clusts = np.array(clusts)
        clusts = np.reshape(clusts, (sum(n_points),2))
        return np.array(clusts)

    def init_dynamics(self, n_clusts, n_points, v, a):
        v0 = []
        a0 = []
        for i in range(n_clusts):
            v_i = self.init_velocity(n_points[i], v[i][0], v[i][1])
            a_i = self.init_acceleration(n_points[i], a[i][0], a[i][1])
            v0.append(v_i)
            a0.append(a_i)
        v0 = np.reshape(np.array(v0), (sum(n_points),2))
        a0 = np.reshape(np.array(a0), (sum(n_points),2))
        return v0, a0

    def init_points(self, n, center, rad):
        points = []
        for node in range(n):
            point = []
            for coordinate in center:
                lb = coordinate - rad
                ub = coordinate + rad
                p = np.random.uniform(low=lb, high=ub, size=1)
                point.append(p)
            points.append(point)
        points = np.reshape(np.array(points), newshape=(n,2))
        return points
    
    def init_velocity(self, n, vx, vy):
        v0 = np.repeat(np.array([[vx, vy]]), repeats=n, axis=0)
        return v0

    def init_acceleration(self, n, ax, ay):
        a0 = np.repeat(np.array([[ax, ay]]), repeats=n, axis=0)
        return a0




ns = NodeSpace()


#%%
# n_clusts = 3
# n_points = [5, 5, 5]
# centers = [[50,-100], [-50,0], [50,50]]
# radius = [5,5,5]
# v = [[0,1], [1,0], [0,-1]]
# a =  [[0,0], [0,0], [0,0]]

# z0 = ns.init_clusters(n_clusts, n_points, centers, radius)
# v0, a0 = ns.init_dynamics(n_clusts, n_points, v, a)
# ns.init_conditions(z0, v0, a0)

n_clusts = 2
n_points = [1, 1]
centers = [[-20,-20], [20,20]]
radius = [0.0,0.0]

v = [[-1,-1], [1,1]]
a =  [[2,2], [1,1]]

z0 = ns.init_clusters(n_clusts, n_points, centers, radius)
v0, a0 = ns.init_dynamics(n_clusts, n_points, v, a)
ns.init_conditions(z0, v0, a0)

# %%
# fig = plt.figure()
# ax = plt.axes(xlim=(-100,100), ylim=(-100,100))

fig, ax = plt.subplots(nrows=2, ncols=1)
fig.tight_layout(pad=3)
ax0 = ax[0]
ax1 = ax[1]
ax0.set_title("Latent space")
ax1.set_title("Intensity with event times")
ax0.set_xlim(-100,100)
ax0.set_ylim(-100,100)
ax1.set_xlim(0, 200)
ax1.set_ylim(0,200)


points, = ax0.plot([], [], "bo", ms=6)
lines, = ax1.plot([],[])

dist_arr = []

x = np.linspace(0,200)

def intensity(euclid_dist):
    b = 1e2
    d = euclid_dist
    return np.exp(b - d)

def lambda_arr(z, v, a, t, b=1e2):
    lambdas = []
    for i in range(len(t)):
        z = z0 + v0*t[i] + 0.5*a0*t[i]**2
        cur_dist = pdist(z, metric="euclidean")
        max_lambda = np.max(np.exp(b - cur_dist))
        lambdas.append(max_lambda)
    return np.array(lambdas)


def init():
    points.set_data([],[])
    lines.set_data([],[])
    return points, lines

def animate(i, x, dist_arr):
    z = z0 + v0*x[i] + 0.5*a0*x[i]**2
    points.set_data(z[:,0], z[:,1])

    H = pdist(z, metric="euclidean")
    
    dist_arr.append(H[0])
    #print(dist_arr)

    lines.set_data(x[0:i], dist_arr[:i])
    return points, lines




if "_name__"=="__main__":
    anim = animation.FuncAnimation(fig, animate, fargs=[x, dist_arr], init_func=init, frames=len(x), interval=50, blit=True)
    #anim
    #print(dir(anim))
    #HTML(anim.to_html5_video())
    #anim
    HTML(anim.to_jshtml())
    anim.save("testvid.mp4", fps=5, extra_args=["-vcodec", "libx264"])

    plt.show()
# %%
