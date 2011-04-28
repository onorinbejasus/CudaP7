// 
//  setup.hh
//  Cloth Simulation
//  
//  Created by Timothy Luciani on 2011-04-10.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
// 

void free_data(void); // cleanup
void draw_particles(void); // draw particles 
void draw_forces(void); // NOT USED
void step_func(); // step forward in time
void init_system(void); // initialize particle system