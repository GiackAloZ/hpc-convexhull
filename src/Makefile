## Makefile di esempio modificato per il progetto di High Performance Computing
## 2019/2020, corso di laurea in Ingegneria e Scienze Informatiche,
## Universita' di Bologna.

## Ultima modifica: 2020-09-27, Giacomo Aloisi <giacomo.aloisi(at)studio.unibo.it>

## Questo e' un frammento del Makefile utilizzato negli esempi
## illustrati durante il corso. Questo file puo' essere modificato e
## personalizzato in base alle proprie esigenze. Si puo' anche
## decidere di non usarlo; in tal caso indicare le istruzioni di
## compilazione nel file README presente nella directory a livello
## superiore.
##
## Se si decide di usare questo makefile, il comando "make" dovrebbe
## compilare tutti i programmi consegnati senza errori né warning.  Si
## consiglia pertanto di rimuovere eventuali target non presenti
## nell'archivio consegnato.
##
## Questo Makefile compila i file "omp-*.c" usando il flag -fopenmp, i
## file "cuda-*.cu" con il compilatore nvcc, e i file "mpi-*.c" con
## mpicc.
##
## I principali target definiti da questo makefile sono:
##
## make         compila tutti i sorgenti disponibili
## make clean   cancella i file temporanei e gli eseguibili
## make openmp  compila la versione OpenMP
## make mpi     compila la versione MPI
## make cuda    compila la versione CUDA
## make images  usa ./convex-hull per creare le immagini di tutti gli inviluppi

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

export COMMAND_EXE:=
export OUTPUT_EXE:=omp-convex-hull

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
