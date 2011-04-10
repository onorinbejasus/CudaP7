# $Id: gfx-config.in 343 2008-09-13 18:34:59Z garland $

CXX = g++
CXXFLAGS = -g -O2 -Wall -Wno-sign-compare -Iinclude -DHAVE_CONFIG_H 
OBJS = Mesh.o stitcher.o

main: $(OBJS)
	$(CXX) -o $@ $^ -lpng -framework GLUT -framework OpenGL -g
clean:
	rm $(OBJS) main
