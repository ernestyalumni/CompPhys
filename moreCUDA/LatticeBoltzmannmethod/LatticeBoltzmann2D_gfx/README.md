# `LatticeBoltzmann2D_gfx`

Single-phase incompressible Navier-Stokes equations solver for `d2p9` with the main loop included in an OpenGL render (rendered directly on the GPU, with CUDA-allocated device GPU-side float arrays, so OpenGL is reading directly from the device GPU!).  

## Parameters to change *MANUALLY*

Tune/Play with these parameters to make it work and print out prettily:

in `main.cu`:
```
N_x,N_y
DENSITY, LID_VELOCITY, REYNOLDS_NUMBER
```
in `./commonlib/tex_anim.2d.cu`,
-   in `__global__ void float_to_char`, *MANUALLY* change `minval`,`maxval`   

-   in `__global__ void float2_to_char`, *MANUALLY* change `minval`,`maxval`

[Youtube video]() (EY: 20170420; it's really pretty to look at, as the dynamics are rendered in real-time, I highly suggest taking a watch).  


