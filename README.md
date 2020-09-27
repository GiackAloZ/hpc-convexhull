# HPC convex hull 2019/2020

Questo archivio contiene le implementazioni e la relazione per il progetto 
del corso di High Performance Computing, laurea in Ingegneria e Scienze
Informatiche, Universita' di Bologna sede di Cesena, AA 2019/2020.

Ho modificato leggermente il Makefile fornito ed ho creato altri Makefile che
vengono invocati ricorsivamente per creare i file di input (quelli ricreabili),
generare gli output `*.hull` e le immagini generate da `plot-hull.gp`.

## Compilazione

Per compilare i file presenti nella cartella `src`, spostarsi nella cartella e
usare il comando `make`:
```sh
path/> cd src
path/src> make
```

Nella cartella `src` sono presenti anche le versioni _no split_ che vengono compilate insieme alle altre usando `make`.

## Generazione file

Per generare i file di input come presente nel Makefile originale, usare:
```sh
path/src> make files
```

Per calcolare tutti gli output dei file `*.in` presenti nella cartella `src/files/inputs`, usare:
```sh
path/src> make outputs
```
Questo comando usa la verisione in OpenMP per calcolare gli output. Si può specificare la versione con:
```sh
path/src> make outputs VER=ver
```
con `ver` a scelta tra `omp` e `mpi`.

Per generare le immagini risultati, usare:
```sh
path/src> make images
```
Anche questo comando accetta la versione.

Si possono inoltre generare gli input utilizzati per il calcolo della weak scaling efficiency muovendosi nella cartella `src/files/inputs/weak-scaling` e usare `make`:
```sh
path/src> cd files/inputs/weak-scaling
path/src/files/inputs/weak-scaling> make
```
