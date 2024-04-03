import numpy as np
import taichi as ti

from image_io import write_ppm
from math_tools import normalize
from taichi_ray import TaichiRay
from rng import RNG
from sampling import cosine_weighted_sample_on_hemisphere
from taichi_sphere import TaichiSphere
from specular import ideal_specular_reflect, ideal_specular_transmit
from sphere import Sphere, Reflection_t
from ray import Ray

ti.init(arch=ti.cpu, default_fp=ti.f64)

EPSILON_SPHERE : ti.f32 = 1e-4
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


@ti.kernel
def intersect(ray: TaichiRay) -> ti.types.vector(2, ti.i32):
    id = -1
    hit = 0
    ti.loop_config(serialize=True)
    for i in range(NUM_SPHERES):
        if taichi_spheres[i].intersect(ray) == 1:
            hit = 1
            id = i
    # if hit:
    #     print(id)
    #     print(ray)
    #     exit(0)
    
    return ti.Vector([hit, id], dt=ti.i32)

def intersectP(ray):
    for i in range(NUM_SPHERES):
        if spheres[i].intersect(ray):
            return True
    return False

def radiance(ray, rng):
    r = ray
    taichi_r = TaichiRay(o=ray.o, d=ray.d, tmin=ray.tmin, tmax=ray.tmax, depth=ray.depth)
    L = np.zeros((3), dtype=np.float64)
    F = np.ones((3), dtype=np.float64)
    while (True):
        hit, id = intersect(taichi_r)
        if (not hit):
            return L

        shape = spheres[id]
        p = r(r.tmax)
        n = normalize(p - shape.p)

        L += F * shape.e
        F *= shape.f
        
	    # Russian roulette
        if r.depth > 4:
            continue_probability = np.amax(shape.f)
            if rng.uniform_float() >= continue_probability:
                return L
            F /= continue_probability

        # Next path segment
        if shape.reflection_t == Reflection_t.SPECULAR:
            d = ideal_specular_reflect(r.d, n)
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue
        elif shape.reflection_t == Reflection_t.REFRACTIVE:
            d, pr = ideal_specular_transmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, rng)
            F *= pr
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue
        else:
            w = n if n.dot(r.d) < 0 else -n
            u = normalize(np.cross(np.array([0.0, 1.0, 0.0], np.float64) if np.fabs(w[0]) > 0.1 else np.array([1.0, 0.0, 0.0], np.float64), w))
            v = np.cross(w, u)

            sample_d = cosine_weighted_sample_on_hemisphere(rng.uniform_float(), rng.uniform_float())
            d = normalize(sample_d[0] * u + sample_d[1] * v + sample_d[2] * w)
            r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            continue

import sys

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
    main()
    # s = TaichiSphere(r=1e5,  p=np.array([1e5 + 1, 40.8, 81.6],    dtype=np.float64), f=np.array([0.75,0.25,0.25],      dtype=np.float64))
    # r = TaichiRay    
