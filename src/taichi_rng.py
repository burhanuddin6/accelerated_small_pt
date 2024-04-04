import taichi as ti

# uniform random number generator in taich

@ti.func
def uniform_float() -> ti.f64:
    return ti.random()