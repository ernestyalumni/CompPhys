/* tex_anim2d.cu
 * 2-dim. GPU texture animation 
 * Ernest Yeung  ernestyalumni@gmail.com
 * 20160720
 */
#include "tex_anim2d.h"
  
int iterationCount = 0 ;

BC bc = {W / 2, H / 2, W / 15.f, 150, 212.f, 70.f, 0.f}; // Boundary conds

// interactions

void keyboard_func( unsigned char key, int x, int y) {
	if (key == 'S') bc.t_s += DT;
	if (key == 's') bc.t_s -= DT;
	if (key == 'A') bc.t_a += DT;
	if (key == 'a') bc.t_a -= DT;
	if (key == 'G') bc.t_g += DT;
	if (key == 'g') bc.t_g -= DT;
	if (key == 'R') bc.rad += DT;
	if (key == 'r') bc.rad = MAX(0.f, bc.rad - DT);
	if (key == 'C') ++bc.chamfer;
	if (key == 'c') --bc.chamfer;

	if (key==27) {
		std::exit(0) ;
	}
	glutPostRedisplay();
}
	
void mouse_func( int button, int state, int x, int y ) {
	bc.x = x, bc.y = y;
	glutPostRedisplay();
}

void idle() {
	++iterationCount;
	glutPostRedisplay();
}

void printInstructions() {
	printf("3 dim. texture animation \n"
			"Relocate source with mouse click\n"
		   "Change source temperature (-/+): s/S\n"
		   "Change air temperature    (-/+): a/A\n"
		   "Change ground temperature (-/+): g/G\n"
		   "Change pipe radius        (-/+): r/R\n"
		   "Change chamfer            (-/+): c/C\n"

			"Exit                           : Esc\n"
	
	);
}

// make* functions make functions to pass into OpenGL (note OpenGL is inherently a C API

void make_draw_texture(int w, int h) {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, 
		GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0,0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0,h);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(w,h);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(w,0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}	
