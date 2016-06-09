# `common` - library or subdirectory that contains header files - documentation/descriptions/explanations/breakdown

cf. Jason Sanders, Edward Kandrot. CUDA by Example: An Introduction to General-Purpose GPU Programming    
[github:jiekebo/CUDA-By-Example/](https://github.com/jiekebo/CUDA-By-Example)

## Dictionary/Index between (header) files on this github subdirectory to code in **CUDA By Example**, *Sanders and Kandrot*

| filename     |   ppg(s).  | (Sub)Section | Description/Uses                  |
| ------------ | :--------: | :----------: | :-------------------------------: |
| cpu_bitmap.h | 46-57      | 4.2.2.       | A Fun Example; Julia Sets         |
| cpu_anim.h   | 70-72-     | 5.2.2.       | GPU Ripple using threads          |


## header file/library breakdown

### `cpu_bitmap.h`
This header file contains 1 struct `CPUBitmap`

*Programming note* in the method `display_and_exit`, inside it, a `char* dummy = "";` is created, pointer to a `char`.  ~~Remember the `const` so you'll never try to to write to this pointer.~~  
cf. [Deprecated conversion from string literal to 'char*'](http://stackoverflow.com/questions/9650058/deprecated-conversion-from-string-literal-to-char)  
The resolution to this was here:  
cf. [How does one represent the empty char?](http://stackoverflow.com/questions/18410234/how-does-one-represent-the-empty-char)  
Representing an empty char as `""` was causing all the problems - doing this `\0` for an empty char made it much better for the compiler.  

### `cpu_anim.h` 
- contains a struct `CPUAnimBitmap`
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



