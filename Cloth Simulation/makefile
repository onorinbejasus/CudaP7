CU_SOURCES = $(shell ls *cu)
CC_SOURCES = $(shell ls *cc)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CC_OBJECTS = $(CC_SOURCES:.cc=.o)

OBJECTS = $(CU_OBJECTS) $(CC_OBJECTS)

OSNAME := $(shell uname -s)
MACHNAME := $(shell uname -n | sed 's/[0-9].*//')
ifeq ($(OSNAME),Darwin)
	include makefile.osx
else
ifeq ($(MACHNAME),ghc)
	include makefile.ghc
else
	include makefile.nix
endif
endif

# this is what we're building
all: main

# dependency handling
.DELETE_ON_ERROR:
-include $(OBJECTS:.o=.d)

# the principal target
main: $(OBJECTS) makefile
	echo objects $(OBJECTS)
	echo cu_objects $(CU_OBJECTS)
	echo cc_objects $(CC_OBJECTS)
	nvcc $(NVCC_CFLAGS) $(NVCC_LFLAGS) -o main -g $(OBJECTS)

$(CU_OBJECTS):
	touch $@
	nvcc $(NVCC_CFLAGS) -c -o $@ $(@:.o=.cu)
	nvcc $(NVCC_CFLAGS) -M $(@:.o=.cu) > $(@:.o=.d)

$(CC_OBJECTS):
	touch $@
	nvcc $(NVCC_CFLAGS) -c -o $@ $(@:.o=.cc)
	nvcc $(NVCC_CFLAGS) -M $(@:.o=.cc) > $(@:.o=.d)

# create source documentation
docs:
	doxygen

# clean the installation
clean:
	rm -rfv main *.o *.d doc
