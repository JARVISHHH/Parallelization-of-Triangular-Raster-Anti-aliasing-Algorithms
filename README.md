# Parallelization-of-Triangular-Raster-Anti-aliasing-Algorithms

Studied the parallelization of two main anti-aliasing algorithms with a classmate to accelerate the anti-aliasing algorithm of triangle rasterization.

### Added the code of the low-pass filter based on the framework code of the GAMES101 (Assignment 2) framework.

The picture before using anti-aliasing algorithms is listed below.
![image](https://user-images.githubusercontent.com/54977500/188248062-05b70622-95e0-4b75-a1a3-1e781295942f.png)

The picture after using  the low-pass filter is listed below.
![image](https://user-images.githubusercontent.com/54977500/188248618-1a441326-d5e4-4ed1-ac6a-50bdedec7663.png)

As we can see, it realized the rasterization after blurring, and the edge had the effect of gradual disappearance. Especially for the blue triangle, the aliasing problem was much reduced.

### Parallelized the Fourier transform and product in the frequency domain in the low-pass filter.

Mainly adopted the method of loop expansion, multithreading, multi-node, and hybrid parallel. Compared with the serial algorithm, the parallelized anti-aliasing algorithm achieved a speed-up ratio up to 7:1.
