import numpy as np

class Reflection_t(object):
    DIFFUSE, SPECULAR, REFRACTIVE = range(3)

class Sphere(object):

    EPSILON_SPHERE = 1e-4
    # r : np.float64 = 0.0 # radius
    # p : np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64) # position
    # e : np.ndarray = np.zeros((3), dtype=np.float64) # emission
    # f : np.ndarray = np.zeros((3), dtype=np.float64) # color
    # reflection_t : Reflection_t = Reflection_t.DIFFUSE # reflection type
    
    def __init__(self, r, p, e = np.zeros((3), dtype=np.float64), f = np.zeros((3), dtype=np.float64), reflection_t = Reflection_t.DIFFUSE):
        self.r = np.float64(r)
        self.p = np.copy(p)
        self.e = np.copy(e)
        self.f = np.copy(f)
        self.reflection_t = reflection_t

    def intersect(self, ray):

        op = self.p - ray.o
        dop = ray.d.dot(op)
        D = dop * dop - op.dot(op) + self.r * self.r

        if D < 0:
            return False

        sqrtD = np.sqrt(D)

        tmin = dop - sqrtD
        if (ray.tmin < tmin and tmin < ray.tmax):
            ray.tmax = tmin
            return True

        tmax = dop + sqrtD
        if (ray.tmin < tmax and tmax < ray.tmax):
            ray.tmax = tmax
            return True
        
        return False
    
    def __str__(self) -> str:
        return 'r: ' + str(self.r) + '\n' + 'p: ' + str(self.p) + '\n' + 'e: ' + str(self.e) + '\n' + 'f: ' + str(self.f) + '\n' + 'reflection_t: ' + str(self.reflection_t) + '\n'