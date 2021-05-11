%matplotlib inline
import numpy as np
import torch
from scipy.spatial.distance import pdist
#from IPython.display import HTML
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
import matplotlib.animation as animation

plt.style.use("seaborn")
plt.rcParams["animation.html"] = "jshtml"

class NodeSpace():
    def __init__(self):
        self.z = None
        self.v = None
        self.a = None
    
    def step(self, dt):
        self.z = self.z[:,:] + self.v[:,:]*dt + 0.5*self.a[:,:]*dt**2

    def init_cond(self, z0, v0, a0):
        self.z = z0
        self.v = v0
        self.a = a0

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

rad = 0
z0_a = ns.init_points(1, [50,-100], rad)
z0_b = ns.init_points(1, [50,100], rad)
z0_c = ns.init_points(1, [50,50], rad)
z0 = np.concatenate((z0_a, z0_b))
#z0 = np.concatenate((z0_a, z0_b, z0_c))

v0_a = ns.init_velocity(1, 0, 1)
v0_b = ns.init_velocity(1, 0, -1)
v0_c = ns.init_velocity(1, 0, -1)
v0 = np.concatenate((v0_a, v0_b))
#v0 = np.concatenate((v0_a, v0_b, v0_c))

#a0_a = ns.init_acceleration(5, 0, -0.01)
#a0_b = ns.init_acceleration(5, -0.01, 0)
#a0_c = ns.init_acceleration(5, 0.01, 0)
a0 = ns.init_acceleration(2, 0, 0)
#a0 = np.concatenate((a0_a,a0_b,a0_c))

ns.init_cond(z0, v0, a0)

# fig = plt.figure()
# ax = plt.axes(xlim=(-100,100), ylim=(-100,100))

fig, ax = plt.subplots(nrows=2, ncols=1)
ax0 = ax[0]
ax1 = ax[1]
ax0.set_xlim(-100,100)
ax0.set_ylim(-100,100)
ax1.set_xlim(0, 100)
ax1.set_ylim(0,100)


points, = ax0.plot([], [], "bo", ms=6)
lines, = ax1.plot([],[])

def init():
    points.set_data([],[])
    lines.set_data([],[])
    return points, lines

def animate(i):
    z = z0 + v0*i + 0.5*a0*i**2
    points.set_data(z[:,0], z[:,1])
    lines.set_data(np.arange(0, i, 1), np.arange(0,i,1))
    return points, lines

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100, blit=True)
#anim
#print(dir(anim))
#HTML(anim.to_html5_video())
anim
#HTML(anim.to_jshtml())
#anim.save("subplot_points.mp4", fps=5, extra_args=["-vcodec", "libx264"])

#plt.show()