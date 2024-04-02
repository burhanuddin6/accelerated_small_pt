# Tools :
import numpy as np
import taichi as ti

def to_byte(x, gamma = 2.2):
    return int(np.clip(255.0 * np.power(x, 1.0 / gamma), a_min=0.0, a_max=255.0))

def write_ppm(w, h, Ls, fname = "numpy-image.ppm"):
    with open(fname, 'w') as outfile:
        outfile.write('P3\n{0} {1}\n{2}\n'.format(w, h, 255));
        for i in range(Ls.shape[0]):
            outfile.write('{0} {1} {2} '.format(to_byte(Ls[i,0]), to_byte(Ls[i,1]), to_byte(Ls[i,2])))

@ti.func
def device_normalize(v):
    norm = ti.math.normalize(v)
    if norm == 0: 
       return v
    return v / norm 

print(normalize(np.arange(10)))