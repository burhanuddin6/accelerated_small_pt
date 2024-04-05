import taichi as ti
import random
ti.init(arch=ti.cpu, default_fp=ti.f64)
# uniform random number generator in taich


@ti.func
def uniform_float() -> ti.f64:
    return random.uniform(0.0, 1.0)


@ti.kernel
def test_uniform_float():
    print(uniform_float())
    
if __name__ == '__main__':
    test_uniform_float()