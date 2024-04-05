from taichi_common import *

@ti.func
def cosine_weighted_sample_on_hemisphere(u1 : ti.f64, u2 : ti.f64) -> ti.math.vec3:
    cos_theta = ti.math.sqrt(1.0 - u1)
    sin_theta = ti.math.sqrt(u1)
    phi = 2.0 * ti.math.pi * u2
    return ti.math.vec3(ti.math.cos(phi) * sin_theta, ti.math.sin(phi) * sin_theta, cos_theta)