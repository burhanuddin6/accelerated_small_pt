from taichi_common import *
from taichi_ray import TaichiRay


@ti.dataclass
class TaichiSphere:
    r : ti.f64
    p : ti.math.vec3
    e : ti.math.vec3
    f : ti.math.vec3
    reflection_t : ti.i32 # 0: DIFFUSE, 1: SPECULAR, 2: REFRACTIVE
    
    @ti.func
    def intersect(self, ray: TaichiRay) -> ti.math.vec2: # returns 0 or 1 (false, true)
        '''
        Please not that the return types are very badly configured
        The first return type should be a bool (or int) and second is a float
        The reason for returning ray.tmax is that taichi is making a copy and not changing
        the value of the variable in place due to which I need to return the value of tmax
        '''
        op = self.p - ray.o
        dop = ti.math.dot(ray.d, op)
        D = (dop * dop) - ti.math.dot(op, op) + (self.r * self.r)

        ret_val = ti.i32(0)
        if D < 0:
            ret_val = 0
        else:
            sqrtD = ti.math.sqrt(D)

            tmin = dop - sqrtD
            if (ray.tmin < tmin and tmin < ray.tmax):
                ray.tmax = tmin
                ret_val = 1
            
            else:
                tmax = dop + sqrtD
                if (ray.tmin < tmax and tmax < ray.tmax):
                    ray.tmax = tmax
                    ret_val = 1
        return ti.math.vec2(ret_val, ray.tmax)
    
    #  @ti.func
    # def intersect(self, ray: TaichiRay) -> bool: # returns 0 or 1 (false, true)
    #     self._intersect

@ti.kernel
def ker() -> ti.math.vec2:
    r = TaichiRay(o=ti.math.vec3(6.07510647, 13.13940276, 167.13795543), d=ti.math.vec3(-0.31104694, -0.27518496, -0.90968293), tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=0)
    x = taichi_spheres[4].intersect(r)
    # this example shows why its important to return tmax
    print("tmax in func: ", x[1])
    print("tmax outside: ", r.tmax)
    return x


if __name__ == '__main__':
    pass
    # print(ker())
    # import ray, numpy_smallpt
    # import numpy as np
    # r = ray.Ray(np.array([6.07510647, 13.13940276, 167.13795543]), np.array([-0.31104694, -0.27518496, -0.90968293]), 0.0, np.inf, 0)
    # print(numpy_smallpt.spheres[4].intersect(r))