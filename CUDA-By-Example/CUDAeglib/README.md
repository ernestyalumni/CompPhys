# `CUDAeglib` documentation/descriptions/explanations/breakdown

cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming    
[github:jiekebo/CUDA-By-Example/](https://github.com/jiekebo/CUDA-By-Example)

## Dictionary/Index between files on this github subdirectory to code in **CUDA By Example**, *Sanders and Kandrot*

| filename   |   ppg(s).  | (Sub)Section | Description/Uses                  |
| ---------- | :--------: | :----------: | :-------------------------------: |
| cpu_anim.h | 70-72-     | 5.2.2.       | GPU Ripple using threads          |


## header file/library breakdown

* `cpu_anim.h` - contains a struct `CPUAnimBitmap`
  - `CPUAnimBitmap` methods
    * `get_ptr`
    * `image_size`
    * `anim_and_exit`
  - `CPUAnimBitmap` static methods (for glut callbacks)
    * `get_bitmap_ptr`
    * `mouse_func`
    * `idle_func`
    * `Key` 
    * `Draw`
  - `cpu_anim.h` has the line `#include "gl_helper.h"`.  Looking at the code for `gl_helper.h`, then for Linux, all that's needed is to include the system's copy of `glut.h, glext.h, glx.h`



