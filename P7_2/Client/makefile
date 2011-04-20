CC_SOURCES = $(shell ls *cpp)
CC_OBJECTS = $(CC_SOURCES:.cpp=.o)

OBJECTS = $(CC_OBJECTS)

OSNAME := $(shell uname -s)
MACHNAME := $(shell uname -n | sed 's/[0-9].*//')
ifeq ($(OSNAME),Darwin)
	include makefile.osx
endif

# this is what we're building
all: main

# dependency handling
.DELETE_ON_ERROR:
-include $(OBJECTS:.o=.d)

# the principal target
main: $(OBJECTS) makefile
	echo objects $(OBJECTS)
	echo cc_objects $(CC_OBJECTS)
	nvcc $(NVCC_CFLAGS) $(NVCC_LFLAGS) -o main -g $(OBJECTS)

$(CC_OBJECTS):
	touch $@
	nvcc $(NVCC_CFLAGS) -c -o $@ $(@:.o=.cpp)
	nvcc $(NVCC_CFLAGS) -M $(@:.o=.cpp) > $(@:.o=.d)

# create source documentation
docs:
	doxygen

# clean the installation
clean:
	rm -rfv main *.o *.d doc
