import taichi as ti
import random
ti.init()
# uniform random number generator in taich


@ti.func
def uniform_float() -> ti.f64:
    return ti.random(float) * ti.random(float)


@ti.kernel
def test_uniform_float():
    print(uniform_float())
    
if __name__ == '__main__':
    test_uniform_float()