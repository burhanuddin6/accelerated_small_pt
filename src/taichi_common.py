import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64)

EPSILON_SPHERE = 1e-4   
DIFFUSE = 0
SPECULAR = 1
REFRACTIVE = 2

# Scene
REFRACTIVE_INDEX_OUT = 1.0
REFRACTIVE_INDEX_IN = 1.5

NUM_SPHERES = 9

