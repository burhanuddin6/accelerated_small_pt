import numpy as np
import taichi as ti
import sys

from image_io import write_ppm
from math_tools import normalize
from rng import RNG
from sphere import Sphere, Reflection_t
from ray import Ray

from taichi_ray import TaichiRay
from taichi_sampling import cosine_weighted_sample_on_hemisphere
from taichi_sphere import TaichiSphere
from taichi_specular import ideal_specular_reflect, ideal_specular_transmit
from taichi_rng import uniform_float

ti.init(arch=ti.cpu, default_fp=ti.f64)

EPSILON_SPHERE = 1e-4   
DIFFUSE = 0
SPECULAR = 1
REFRACTIVE = 2

# Scene
REFRACTIVE_INDEX_OUT = 1.0
REFRACTIVE_INDEX_IN = 1.5


spheres = [
        Sphere(r=1e5,  p=np.array([1e5 + 1, 40.8, 81.6],    dtype=np.float64), f=np.array([0.75,0.25,0.25],      dtype=np.float64)),
	    Sphere(r=1e5,  p=np.array([-1e5 + 99, 40.8, 81.6],  dtype=np.float64), f=np.array([0.25,0.25,0.75],      dtype=np.float64)),
	    Sphere(r=1e5,  p=np.array([50, 40.8, 1e5],          dtype=np.float64), f=np.array([0.75, 0.75, 0.75],    dtype=np.float64)),
	    Sphere(r=1e5,  p=np.array([50, 40.8, -1e5 + 170],   dtype=np.float64)),
	    Sphere(r=1e5,  p=np.array([50, 1e5, 81.6],          dtype=np.float64), f=np.array([0.75, 0.75, 0.75],    dtype=np.float64)),
	    Sphere(r=1e5,  p=np.array([50, -1e5 + 81.6, 81.6],  dtype=np.float64), f=np.array([0.75, 0.75, 0.75],    dtype=np.float64)),
	    Sphere(r=16.5, p=np.array([27, 16.5, 47],           dtype=np.float64), f=np.array([0.999, 0.999, 0.999], dtype=np.float64), reflection_t=Reflection_t.SPECULAR),
	    Sphere(r=16.5, p=np.array([73, 16.5, 78],           dtype=np.float64), f=np.array([0.999, 0.999, 0.999], dtype=np.float64), reflection_t=Reflection_t.REFRACTIVE),
	    Sphere(r=600,  p=np.array([50, 681.6 - .27, 81.6],  dtype=np.float64), e=np.array([12, 12, 12],          dtype=np.float64))
        ]

NUM_SPHERES = 8
taichi_spheres = TaichiSphere.field(shape=(8,))
taichi_spheres[0] = TaichiSphere(r=1e5,  p=ti.math.vec3(1e5 + 1, 40.8, 81.6), e=ti.math.vec3(0), f=ti.math.vec3(0.75,0.25,0.25), reflection_t=DIFFUSE)
taichi_spheres[1] = TaichiSphere(r=1e5,  p=ti.math.vec3(-1e5 + 99, 40.8, 81.6), e=ti.math.vec3(0), f=ti.math.vec3(0.25,0.25,0.75), reflection_t=DIFFUSE)
taichi_spheres[2] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, 40.8, 1e5), e=ti.math.vec3(0), f=ti.math.vec3(0.75, 0.75, 0.75), reflection_t=DIFFUSE)
taichi_spheres[3] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, 40.8, -1e5 + 170), e=ti.math.vec3(0), f=ti.math.vec3(0), reflection_t=DIFFUSE)
taichi_spheres[4] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, 1e5, 81.6), e=ti.math.vec3(0), f=ti.math.vec3(0.75, 0.75, 0.75), reflection_t=DIFFUSE)
taichi_spheres[5] = TaichiSphere(r=1e5,  p=ti.math.vec3(50, -1e5 + 81.6, 81.6), e=ti.math.vec3(0), f=ti.math.vec3(0.75, 0.75, 0.75), reflection_t=DIFFUSE)
taichi_spheres[6] = TaichiSphere(r=16.5, p=ti.math.vec3(27, 16.5, 47), e=ti.math.vec3(0), f=ti.math.vec3(0.999, 0.999, 0.999), reflection_t=SPECULAR)
taichi_spheres[7] = TaichiSphere(r=16.5, p=ti.math.vec3(73, 16.5, 78), e=ti.math.vec3(0), f=ti.math.vec3(0.999, 0.999, 0.999), reflection_t=REFRACTIVE)
taichi_spheres[8] = TaichiSphere(r=600,  p=ti.math.vec3(50, 681.6 - .27, 81.6), e=ti.math.vec3(12, 12, 12), f=ti.math.vec3(0), reflection_t=DIFFUSE)


@ti.func
def intersect(ray: TaichiRay) -> ti.types.vector(3, ti.f64):
    '''
    Bad return type
    Its not [float float float] but [int int float] actually but bootstrapping for now
    '''
    id = -1
    hit = 0
    tmax = ti.math.inf
    for i in range(NUM_SPHERES):
        hit_tmax = taichi_spheres[i].intersect(ray)
        if hit_tmax[0] == 1:
            tmax = hit_tmax[1]
            hit = 1
            id = i

    return ti.Vector([hit, id, tmax], dt=ti.f64)


@ti.func
def radiance(ray: TaichiRay) -> ti.types.vector(3, ti.f64):
    ret_val = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    r = ray
    L = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    F = ti.Vector([1.0, 1.0, 1.0], dt=ti.f64)
    
    # loop variables, defining them above for efficiency (?)
    n = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    w = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    v = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    u = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    while True:
        hit_id_tmax = intersect(r)
        hit = int(hit_id_tmax[0])
        id = int(hit_id_tmax[1])
        tmax = hit_id_tmax[2]
        r.tmax = tmax
        if not hit:
            ret_val = L
            break

        shape = taichi_spheres[id]
        print(r.tmax)
        p = r.at(r.tmax)
        print("p: ", p, "r.tmax: ", r.tmax)
        print("id: ", id)
        # n = normalize(p - shape.p)
        n = ti.math.normalize(p - shape.p)
        
        L += F * shape.e
        F *= shape.f

        # Russian roulette
        if r.depth > 4:
            continue_probability = ti.math.max(shape.f[0], shape.f[1], shape.f[2])
            random_val = ti.random(float)
            if  random_val >= continue_probability:
                ret_val = L
                break
            F /= continue_probability

        # Next path segment
        if shape.reflection_t == DIFFUSE:
            # if n.dot(r.d) < 0:
            if ti.math.dot(n, r.d) < 0:
                w = n  
            else:
                w = -n
                
            # u = normalize(ti.cross(ti.vec3(0.0, 1.0, 0.0) if ti.abs(w[0]) > 0.1 else ti.vec3(1.0, 0.0, 0.0), w)
            # np.array([0.0, 1.0, 0.0], np.float64) if np.fabs(w[0]) > 0.1 else np.array([1.0, 0.0, 0.0], np.float64)
            if ti.abs(w[0]) > 0.1:
                u = ti.math.normalize(ti.math.cross(ti.Vector([0.0, 1.0, 0.0], dt=ti.f64), w))
            else:
                u = ti.math.normalize(ti.math.cross(ti.Vector([1.0, 0.0, 0.0], dt=ti.f64), w))
            v = ti.math.cross(w, u)

            sample_d = cosine_weighted_sample_on_hemisphere(uniform_float(), uniform_float())
            # d = normalize(sample_d[0] * u + sample_d[1] * v + sample_d[2] * w)
            d = ti.math.normalize(sample_d[0] * u + sample_d[1] * v + sample_d[2] * w)
            r = TaichiRay(o=p, d=d, tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=r.depth + 1)
        elif shape.reflection_t == SPECULAR:
            d = ideal_specular_reflect(r.d, n)
            r = TaichiRay(o=p, d=d, tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=r.depth + 1)
        else:
            # d, pr = ideal_specular_transmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN)
            temp_vec4 = ideal_specular_transmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN)
            d = ti.Vector([temp_vec4[0], temp_vec4[1], temp_vec4[2]], dt=ti.f64)
            pr = temp_vec4[3]
            
            F *= pr
            r = TaichiRay(o=p, d=d, tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=r.depth + 1)
    return ret_val
            

@ti.kernel
def test_taichi_radiance():
    ray = TaichiRay(o=ti.Vector([7.88639855, 13.12551749, 167.13854711]), d=ti.Vector([-0.29938568, -0.27635878, -0.91323274]), tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=0)
    print(radiance(ray))
    
@ti.kernel
def test_taichi_intersect():
    ray = TaichiRay(o=ti.math.vec3(6.0158059,  13.1736803, 167.1364948), d=ti.math.vec3(-0.31144403, -0.27492206, -0.90962656), tmin=0.0, tmax=ti.math.inf, depth=0)
    print(intersect(ray))

def main():
    rng = RNG()
    nb_samples = int(sys.argv[1]) // 4 if len(sys.argv) > 1 else 1

    w = 100
    h = 100

    eye = np.array([50, 52, 295.6], dtype=np.float64)
    gaze = normalize(np.array([0, -0.042612, -1], dtype=np.float64))
    fov = 0.5135
    cx = np.array([w * fov / h, 0.0, 0.0], dtype=np.float64)
    cy = normalize(np.cross(cx, gaze)) * fov

    Ls = np.zeros((w * h, 3), dtype=np.float64)
    
    for y in range(h):
        # pixel row
        print('\rRendering ({0} spp) {1:0.2f}%'.format(nb_samples * 4, 100.0 * y / (h - 1)))
        for x in range(w):
            # pixel column
            for sy in range(2):
                i = (h - 1 - y) * w + x
                # 2 subpixel row
                for sx in range(2):
                    # 2 subpixel column
                    L = np.zeros((3), dtype=np.float64)
                    for s in range(nb_samples):
                        #  samples per subpixel
                        u1 = 2.0 * rng.uniform_float()
                        u2 = 2.0 * rng.uniform_float()
                        dx = np.sqrt(u1) - 1.0 if u1 < 1 else 1.0 - np.sqrt(2.0 - u1)
                        dy = np.sqrt(u2) - 1.0 if u2 < 1 else 1.0 - np.sqrt(2.0 - u2)
                        d = cx * (((sx + 0.5 + dx) / 2.0 + x) / w - 0.5) + \
                            cy * (((sy + 0.5 + dy) / 2.0 + y) / h - 0.5) + gaze
                        L += radiance(Ray(eye + d * 130, normalize(d), tmin=Sphere.EPSILON_SPHERE), rng) * (1.0 / nb_samples)
                    Ls[i,:] += 0.25 * np.clip(L, a_min=0.0, a_max=1.0)

    write_ppm(w, h, Ls)

if __name__ == "__main__":
    # test_taichi_intersect()
    test_taichi_radiance()
