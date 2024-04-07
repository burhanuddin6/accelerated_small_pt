# ACCELERATED SMALL PATH TRACER
This is accelerated version of the famous smallpt code. The original code can be found [here](http://www.kevinbeason.com/smallpt/). This repository uses the [python-numpy](https://github.com/matt77hias/numpy-smallpt) implementation of the original code and uses taichi to accelerate the rendering process. The results are quite impressive. The speed up is at least more than 30x compared to the original code. The code is written in a way that it can be easily modified to add more features. Currently the parallelization is done on the individual pixels but can be easily modified to parallelize on the rays (each sample of each pixel).

## Sample Run
This is the original cornwell scene taken from smallpt. This image was rendered using 512 samples per pixel and 1024x768 resolution. The rendering takes around 2 minutes on a i7-5500U CPU. The image was converted to jpg from ppm format.

![Sample Image](./docs/taichi-cornwell.jpg)