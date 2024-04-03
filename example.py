import taichi as ti
import math
ti.init(arch=ti.cpu)
vec3 = ti.math.vec3

@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32

    @ti.func
    def area(self):
        # a function to run in taichi scope
        return 4 * math.pi * self.radius * self.radius

    def is_zero_sized(self):
        # a python scope function
        return self.radius == 0.0
    
    
a_python_struct = Sphere(center=ti.math.vec3(0.0), radius=1.0)
# calls a python scope function from python
a_python_struct.is_zero_sized() # False

@ti.kernel
def get_area() -> ti.f32:
    a_taichi_struct = Sphere(center=ti.math.vec3(0.0), radius=4.0)
    # return the area of the sphere, a taichi scope function
    return a_taichi_struct.area()

print(get_area()) # 201.062...