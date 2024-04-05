import numpy as np
import sys
import time

from image_io import write_ppm
# from math_tools import normalize
# from rng import RNG
# from sphere import Sphere, Reflection_t
# from ray import Ray

from taichi_common import *
from taichi_ray import TaichiRay
from taichi_sampling import cosine_weighted_sample_on_hemisphere
from taichi_sphere import TaichiSphere
from taichi_specular import ideal_specular_reflect, ideal_specular_transmit
from taichi_rng import uniform_float

if(sys.argv[2] == "gpu"):
    ti.init(arch=ti.gpu, default_fp=ti.f64, random_seed=606418532)
else:
    ti.init(arch=ti.cpu, default_fp=ti.f64, random_seed=606418532)

taichi_spheres = TaichiSphere.field(shape=(NUM_SPHERES,))
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
    ti.loop_config(serialize=True)
    for i in range(NUM_SPHERES):
        hit_tmax = taichi_spheres[i].intersect(ray)
        if hit_tmax[0] == 1:
            # print("id: ", i, "ray.tmax taichi: ", hit_tmax[1])
            tmax = hit_tmax[1]
            ray.tmax = tmax
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
    
#     ti.loop_config(serialize=True)
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
        # print(r.tmax)
        p = r.at(r.tmax)
        # print("p: ", p, "r.tmax: ", r.tmax)
        # print("id: ", id)
        # n = normalize(p - shape.p)
        n = ti.math.normalize(p - shape.p)
        
        L += F * shape.e
        F *= shape.f
        # print("L:", L)

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
            # w = n if n.dot(r.d) < 0 else -n
            # u = normalize(np.cross(np.array([0.0, 1.0, 0.0], np.float64) if np.fabs(w[0]) > 0.1 else np.array([1.0, 0.0, 0.0], np.float64), w))
            # v = np.cross(w, u)

            # sample_d = cosine_weighted_sample_on_hemisphere(rng.uniform_float(), rng.uniform_float())
            # d = normalize(sample_d[0] * u + sample_d[1] * v + sample_d[2] * w)
            # r = Ray(p, d, tmin=Sphere.EPSILON_SPHERE, depth=r.depth + 1)
            # continue
            # if n.dot(r.d) < 0:
            if ti.math.dot(n, r.d) < 0:
                w = n  
            else:
                w = -n
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
    r = TaichiRay(o=ti.math.vec3(6.07510647, 13.13940276, 167.13795543), d=ti.math.vec3(-0.31104694, -0.27518496, -0.90968293), tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=0)
    print(radiance(r))
    
@ti.kernel
def test_taichi_intersect():
    ray = TaichiRay(o=ti.math.vec3(6.0158059,  13.1736803, 167.1364948), d=ti.math.vec3(-0.31144403, -0.27492206, -0.90962656), tmin=0.0, tmax=ti.math.inf, depth=0)
    print(intersect(ray))

@ti.func
def clip(v, a_min, a_max):
    return ti.Vector([
        ti.min(ti.max(v[0], a_min), a_max),
        ti.min(ti.max(v[1], a_min), a_max),
        ti.min(ti.max(v[2], a_min), a_max)
    ])

@ti.kernel
def main(nb_samples: int, w: int, h: int):    

    eye = ti.Vector([50, 52, 295.6], dt=ti.f64)
    gaze = ti.math.normalize(ti.Vector([0, -0.042612, -1], dt=ti.f64))
    fov = 0.5135
    cx = ti.Vector([w * fov / h, 0.0, 0.0], dt=ti.f64)
    cy = ti.math.normalize(ti.math.cross(cx, gaze)) * fov

    
    for y,x in ti.ndrange(h, w):
        # pixel row and column
        # print('\rRendering ({0} spp) {1:0.2f}%'.format(nb_samples * 4, 100.0 * y / (h - 1)))                    
        for sy in range(2):
            i = (h - 1 - y) * w + x
            # 2 subpixel row
            for sx in range(2):
                # 2 subpixel column
                L = ti.math.vec3(0)
                for s in range(nb_samples):
                    #  samples per subpixel
                    u1 = 2.0 * uniform_float()
                    u2 = 2.0 * uniform_float()
                    dx = ti.math.sqrt(u1) - 1.0 if u1 < 1 else 1.0 - ti.math.sqrt(2.0 - u1)
                    dy = ti.math.sqrt(u2) - 1.0 if u2 < 1 else 1.0 - ti.math.sqrt(2.0 - u2)
                    d = cx * (((sx + 0.5 + dx) / 2.0 + x) / w - 0.5) + \
                        cy * (((sy + 0.5 + dy) / 2.0 + y) / h - 0.5) + gaze
                    temp = radiance(TaichiRay(o=eye + d * 130, d=ti.math.normalize(d), tmin=EPSILON_SPHERE, tmax=ti.math.inf, depth=0)) 
                    L += temp * (1.0 / nb_samples)
                    # print("radiance:", temp)                
                temp = 2 * clip(L, a_min=0.0, a_max=1.0)
                Ls[i,0] += temp[0]
                Ls[i,1] += temp[1]
                Ls[i,2] += temp[2]

if __name__ == "__main__":
    elapsed_time = time.time() 

    w = 1024
    h = 768
    Ls = ti.field(dtype=ti.f64, shape=(w * h, 3))
    nb_samples = int(sys.argv[1]) // 4 if len(sys.argv) > 1 else 1
    main(nb_samples, w, h)
    ti.sync()

    elapsed_time = time.time() - elapsed_time
    with open("time.txt", "a") as file:
        file.write(str(elapsed_time))
    print(elapsed_time)
    write_ppm(w, h, Ls, "taichi-image.ppm")
