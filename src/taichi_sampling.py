import taichi as ti

@ti.func
def uniform_sample_on_hemisphere(u1, u2):
    sin_theta = ti.math.sqrt(ti.math.max(0.0, 1.0 - u1 * u1))
    phi = 2.0 * ti.math.pi * u2
    return ti.math.vec3(ti.math.cos(phi) * sin_theta, ti.math.sin(phi) * sin_theta, u1)

@ti.func
def cosine_weighted_sample_on_hemisphere(u1, u2):
    cos_theta = ti.math.sqrt(1.0 - u1)
    sin_theta = ti.math.sqrt(u1)
    phi = 2.0 * ti.math.pi * u2
    return ti.math.vec3(ti.math.cos(phi) * sin_theta, ti.math.sin(phi) * sin_theta, cos_theta)