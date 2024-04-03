import taichi as ti
from taichi_ray import TaichiRay

EPSILON_SPHERE : ti.f32 = 1e-4

DIFFUSE = 0
SPECULAR = 1
REFRACTIVE = 2

ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.dataclass
class TaichiSphere:
    r : ti.f64
    p : ti.math.vec3
    e : ti.math.vec3
    f : ti.math.vec3
    reflection_t : ti.i32 # 0: DIFFUSE, 1: SPECULAR, 2: REFRACTIVE
    
    @ti.func
    def intersect(self, ray: TaichiRay) -> ti.i32: # returns 0 or 1 (false, true)
        op = self.p - ray.o
        dop = ti.math.dot(ray.d, op)
        D = dop * dop - ti.math.dot(op, op) + self.r * self.r

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
            
        return ret_val
    
    #  @ti.func
    # def intersect(self, ray: TaichiRay) -> bool: # returns 0 or 1 (false, true)
    #     self._intersect


spheres = TaichiSphere.field(shape=(8,))
spheres[0] = TaichiSphere(r=1e5,  p=ti.math.vec3(-1e5 + 99, 40.8, 81.6), e=ti.math.vec3(0), f=ti.math.vec3(0.25,0.25,0.75), reflection_t=DIFFUSE)
spheres[1] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, 40.8, 1e5), e=ti.math.vec3(0), f=ti.math.vec3(0.75, 0.75, 0.75), reflection_t=DIFFUSE)
spheres[2] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, 40.8, -1e5 + 170), e=ti.math.vec3(0), f=ti.math.vec3(0), reflection_t=DIFFUSE)
spheres[3] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, 1e5, 81.6), e=ti.math.vec3(0), f=ti.math.vec3(0.75, 0.75, 0.75), reflection_t=DIFFUSE)
spheres[4] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, -1e5 + 81.6, 81.6), e=ti.math.vec3(0), f=ti.math.vec3(0.75, 0.75, 0.75), reflection_t=DIFFUSE)
spheres[5] = TaichiSphere(r=16.5, p=ti.math.vec3(27, 16.5, 47), e=ti.math.vec3(0), f=ti.math.vec3(0.999, 0.999, 0.999), reflection_t=SPECULAR)
spheres[6] = TaichiSphere(r=16.5, p=ti.math.vec3(73, 16.5, 78), e=ti.math.vec3(0), f=ti.math.vec3(0.999, 0.999, 0.999), reflection_t=REFRACTIVE)
spheres[7] = TaichiSphere(r=600,  p=ti.math.vec3(50, 681.6 - .27, 81.6), e=ti.math.vec3(12, 12, 12), f=ti.math.vec3(0), reflection_t=DIFFUSE)

@ti.kernel
def ker() -> ti.i32:
    r = TaichiRay(o=ti.math.vec3(5.51343103, 13.12694412, 167.13848632), d=ti.math.vec3(-0.31462631, -0.27492536, -0.90852976), tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=0)
    return spheres[0].intersect(r)

if __name__ == '__main__':
    print(ker())