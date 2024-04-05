import numpy as np

from image_io import write_ppm
from math_tools import normalize
from ray import Ray
from rng import RNG
from sampling import cosine_weighted_sample_on_hemisphere
from sphere import Sphere, Reflection_t
from specular import ideal_specular_reflect, ideal_specular_transmit

# Scene
REFRACTIVE_INDEX_OUT = 1.0
REFRACTIVE_INDEX_IN = 1.5

spheres = [
    Sphere(r=1e5, p=np.array([50.0, 1e5 - 4.0, 81.6]), e=np.array([0.0, 0.0, 0.0]), f=np.array([1.0, 1.0, 1.0]), reflection_t=Reflection_t.DIFFUSE),  # Botm
    Sphere(r=12.0, p=np.array([48.0, 32.0, 24.0]), e=np.array([3.0, 3.0, 3.0]), f=np.array([0.0, 0.0, 0.0]), reflection_t=Reflection_t.DIFFUSE),  # light
    Sphere(r=12.0, p=np.array([24.0, 8.0, 40.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([0.408, 0.741, 0.467]), reflection_t=Reflection_t.DIFFUSE),  # small sphere 2
    Sphere(r=12.0, p=np.array([24.0, 8.0, -8.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([0.392, 0.584, 0.929]), reflection_t=Reflection_t.DIFFUSE),  # 3
    Sphere(r=12.0, p=np.array([20.0, 52.0, 40.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([1.0, 0.498, 0.314]), reflection_t=Reflection_t.DIFFUSE),  # 5
    Sphere(r=12.0, p=np.array([24.0, 48.0, -8.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([0.95, 0.95, 0.95]), reflection_t=Reflection_t.SPECULAR),  # 5
    Sphere(r=12.0, p=np.array([72.0, 8.0, 40.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([0.95, 0.95, 0.95]), reflection_t=Reflection_t.SPECULAR),  # 3
    Sphere(r=12.0, p=np.array([72.0, 8.0, -8.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([1.0, 0.498, 0.314]), reflection_t=Reflection_t.DIFFUSE),  # 2
    Sphere(r=12.0, p=np.array([76.0, 52.0, 40.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([0.392, 0.584, 0.929]), reflection_t=Reflection_t.DIFFUSE),  # 1
    Sphere(r=12.0, p=np.array([72.0, 48.0, -8.0]), e=np.array([0.0, 0.0, 0.0]), f=np.array([0.408, 0.741, 0.467]), reflection_t=Reflection_t.DIFFUSE)
]

def intersect(ray):
    id = None
    hit = False
    for i in range(len(spheres)):
        if spheres[i].intersect(ray):
            # print("id: ", i, "ray.tmax np: ", ray.tmax)
            hit = True
            id = i
    return hit, id

def intersectP(ray):
    for i in range(len(spheres)):
        if spheres[i].intersect(ray):
            return True
    return False

def radiance(ray: Ray, rng: RNG):
    r = ray
    L = np.zeros((3), dtype=np.float64)
    F = np.ones((3), dtype=np.float64)
    while (True):
        # print("ray.o=", r.o, "ray.d=", r.d)
        hit, id = intersect(r)
        if (not hit):
            return L

        shape = spheres[id]
        p = r(r.tmax)
        # print("p: ", p, "r.tmax: ", r.tmax)
        # print("id: ", id)
        n = normalize(p - shape.p)

        L += F * shape.e
        # print("L: ", L, "shape.e: ", shape.e)
        F *= shape.f
        # print("L:", L)
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
            continue #0.13849026 -0.62708959  0.76653708
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

    w = 400
    h = 400

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
                        temp = radiance(Ray(eye + d * 130, normalize(d), tmin=Sphere.EPSILON_SPHERE), rng) * (1.0 / nb_samples)
                        L += temp
                        # print('==================================================')
                        # print("ray: ", Ray(eye + d * 130, normalize(d), tmin=Sphere.EPSILON_SPHERE))
                        # print(temp)
                        # if all(i > 0 for i in temp):
                        #     exit(0)
                    
                    Ls[i,:] += 0.25 * np.clip(L, a_min=0.0, a_max=1.0)

    write_ppm(w, h, Ls, "numpy_nineball.ppm")

if __name__ == "__main__":
    main()
    # s = Sphere(r=1e5,  p=np.array([1e5 + 1, 40.8, 81.6],    dtype=np.float64), f=np.array([0.75,0.25,0.25],      dtype=np.float64)
