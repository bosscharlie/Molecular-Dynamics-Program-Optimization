CPP = mpiicpc 
OPT = -qopenmp
CFLAGS =  $(OPT)
LDFLAGS =  -qopenmp 
LDLIBS = $(LDFLAGS)
targets = md.exe
objects = md.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

md.exe : md.o
	$(CPP) -o $@ $^ $(LDLIBS)

md.o : md.cpp
	$(CPP) -c $(CFLAGS) $< -o $@


.PHONY: clean
clean:
	rm -rf $(targets) $(objects)

run:
	bsub    -I -q q_x86_vio_share  -N 8 -np 2  ./md.exe  163840  100    2>&1 | tee log 
