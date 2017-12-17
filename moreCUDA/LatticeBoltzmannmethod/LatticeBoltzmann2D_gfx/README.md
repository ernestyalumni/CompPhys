*abridged version*  

`main.cu`  

```  
#include "./physlib/dev_R2grid.h"  	// Dev_Grid2d
#include "./physlib/init.h"			// set_u_0_CPU, dev_e[9], dev_alpha[9], dev_ant[9]
#include "./physlib/PDE.h"			// timeIntegration

#include "./commonlib/tex_anim2d.h" // GPUAnim2dTex

...

#include <functional> // std::bind
```  



