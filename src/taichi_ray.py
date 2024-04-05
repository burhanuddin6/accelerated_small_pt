# import numpy as np

# class Ray(object):

#     def __init__(self, o, d, tmin = 0.0, tmax = np.inf, depth = 0):
#         self.o = np.copy(o)
#         self.d = np.copy(d)
#         self.tmin = tmin
#         self.tmax = tmax
#         self.depth = depth

#     def __call__(self, t):
#         return self.o + self.d * t

#     def __str__(self):
#         return 'o: ' + str(self.o) + '\n' + 'd: ' + str(self.d) + '\n'

import taichi as ti

@ti.dataclass
class TaichiRay:
    o : ti.math.vec3
    d : ti.math.vec3
    tmin : ti.f64
    tmax : ti.f64 
    depth : ti.i32
    
    @ti.func
    def at(self, t: ti.f64) -> ti.math.vec3:
        return self.o + self.d * t


    # @ti.func
    # def __str__(self):
    #     return 'o: ' + str(self.o) + '\n' + 'd: ' + str(self.d) + '\n'

@ti.kernel
def test_ray():
    ray = TaichiRay(ti.Vector([0.0, 0.0, 0.0]), ti.Vector([1.0, 1.0, 1.0]), 0.0, 1.0, 0)
    print(ray.at(0.5))
    