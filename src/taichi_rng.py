import taichi as ti
import random


@ti.func
def uniform_float() -> ti.f64:
    return ti.randn(dt=ti.f64)


@ti.kernel
def test_uniform_float():
    print(uniform_float())
    
if __name__ == '__main__':
    test_uniform_float()