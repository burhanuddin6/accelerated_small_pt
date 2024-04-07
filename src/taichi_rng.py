import taichi as ti
import numpy as np

ti.init(arch=ti.cpu, default_fp=ti.f64)

NUM_RANDOM_NUMBERS = 100000
SEED = 606418532
INDEX = ti.field(dtype=ti.i32, shape=())
INDEX[None] = 0
RANDOM_NUMBERS = ti.field(dtype=ti.f64, shape=NUM_RANDOM_NUMBERS)

@ti.func
def uniform_float() -> ti.f64:
    INDEX[None] = (INDEX[None] + 1) % NUM_RANDOM_NUMBERS
    return RANDOM_NUMBERS[INDEX[None] - 1]

def generate_random():
    np.random.seed(SEED)
    file = open("random.txt", "w")
    for i in range(NUM_RANDOM_NUMBERS):
        file.write(str(np.random.random()) + "\n")
    file.close()
    
def read_random():
    with open("random.txt", "r") as file:
        for i, line in enumerate(file):
            RANDOM_NUMBERS[i] = float(line)


            

@ti.kernel
def test_uniform_float():
    print(uniform_float())
    
if __name__ == '__main__':
    generate_random()
    