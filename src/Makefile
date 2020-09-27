## Makefile di esempio modificato per il progetto di High Performance Computing
## 2019/2020, corso di laurea in Ingegneria e Scienze Informatiche,
## Universita' di Bologna.

## Ultima modifica: 2020-09-27, Giacomo Aloisi <giacomo.aloisi(at)studio.unibo.it>

EXE_OMP:=$(basename $(wildcard omp-*.c))
EXE_MPI:=$(basename $(wildcard mpi-*.c))
EXE_SERIAL:=convex-hull

FILES_DIR:=files

EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL)
CFLAGS+=-std=c99 -Wall -Wpedantic -O2 -D_XOPEN_SOURCE=600
LDLIBS+=-lm
NVCC:=nvcc
NVCFLAGS+=-Wno-deprecated-gpu-targets
NVLDLIBS+=-lm

ifndef $(VER)
VER:=omp
endif

ifeq ($(VER), omp)
export COMMAND_EXE:=
export OUTPUT_EXE:=omp-convex-hull
endif

ifeq ($(VER), mpi)
export COMMAND_EXE:=mpirun
export OUTPUT_EXE:=mpi-convex-hull
endif

.PHONY: clean inputs outputs images

all: $(EXE)

inputs:
	$(MAKE) inputs --directory $(FILES_DIR)

outputs: $(OUTPUT_EXE) inputs
	$(MAKE) outputs --directory $(FILES_DIR)

images: outputs
	$(MAKE) images --directory $(FILES_DIR)

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o bin/$@ $(NVLDLIBS)

$(EXE_OMP): CFLAGS+=-fopenmp
$(EXE_OMP): LDLIBS+=-lgomp
openmp: $(EXE_OMP)

$(EXE_MPI): CC=mpicc
mpi: $(EXE_MPI)

clean:
	$(MAKE) clean --directory $(FILES_DIR)
	\rm -f $(EXE) *.o *~
