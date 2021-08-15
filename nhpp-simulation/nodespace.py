import numpy as np
import scipy.spatial.distance as sd

class NodeSpace():
    def __init__(self):
        self.beta = 5.0
        self.z0 = None
        self.v0 = None
        self.a0 = None
        self.z = None
        self.v = None
        self.a = None
    
    def step(self, t):
            self.z = self.z0[:,:] + self.v0[:,:]*t + 0.5*self.a0[:,:]*t**2
            return self.z
    
    def lambda_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_dist(t, u, v)
        l = np.exp(self.beta - d)
        return np.around(l, decimals=10)

    def lambda_ddt(self, t, u, v):
        z = self.step(t)
        dist = self.get_dist(t, u, v)
        z_uv = z[u,:] - z[v,:]
        z_uv_ddt = (self.v0[u,:]-self.v0[v,:]) + (self.a0[u,:]- self.a0[v,:])*t
        return -np.exp(self.beta - dist)*np.dot(z_uv, z_uv_ddt) / dist

    def lambda_int_rapprox(self, t, u, v):
        dt = np.mean(t[1:len(t)]-t[0:len(t)-1])
        rsum = 0
        for t_i in t:
            l = self.lambda_fun(t_i, u, v)
            rsum += dt * l
        return rsum
    
    def lambda_int_sq_rapprox(self, t, u, v):
        dt = np.mean(t[1:len(t)]-t[0:len(t)-1])
        rsum = 0
        for t_i in t:
            l = self.lambda_sq_fun(t_i, u, v)
            rsum += dt * l
        return rsum

    def lambda_sq_fun(self, t, u, v):
        z = self.step(t)
        d = self.get_sq_dist(t, u, v)
        l = np.exp(self.beta - d)
        return np.around(l, decimals=10)

    def lambda_sq_ddt(self, t, u, v):
        z = self.step(t)
        dist = self.get_sq_dist(t, u, v)
        z_uv = z[u,:] - z[v,:]
        z_uv_ddt = (self.v0[u,:]-self.v0[v,:]) + (self.a0[u,:]- self.a0[v,:])*t
        return -np.exp(self.beta - dist)*np.dot(z_uv, z_uv_ddt)

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

    def rand_init_dynamics(self, n_points):
        v0 = np.random.uniform(-1, 1, size=(n_points, 2))
        a0 = np.random.uniform(-1, 1, size=(n_points, 2))
        return v0, a0

    def custom_init_dynamics(self, n_points, labels, vdir, adir):
        v0 = np.zeros(shape=(n_points, 2))
        a0 = np.zeros(shape=(n_points, 2))

        for idx, label in enumerate(labels):
            #noise = np.random.normal(loc=0.0, scale=1e-3, size=(2,2))
            noise = np.zeros(shape=(2,2))
            v0[idx,:] = vdir[label] + noise[0]
            a0[idx,:] = adir[label] + noise[1]

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
    
    def get_dist(self, t, u, v):
        m = len(self.z0)
        if u==v: 
            return 0.0
        if u > v:
            u, v = v, u
        z = self.step(t)
        d = sd.pdist(z, metric="euclidean")
        idx = m * u + v - ((u + 2) * (u + 1)) // 2
        return d[idx]
    
    def get_sq_dist(self, t, u, v):
        dist = self.get_dist(t,u,v)
        sqdist = dist**2
        debug=True
        return self.get_dist(t,u,v)**2