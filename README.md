Questo archivio contiene il materiale per il progetto del corso di
High Performance Computing, laurea in Ingegneria e Scienze
Informatiche, Universita' di Bologna sede di Cesena, AA 2019/2020.

Il file README (questo file) dovra' includere le istruzioni per la
compilazione e l'esecuzione dei programmi consegnati; per comodita',
nella directory src/ e' presente un Makefile che dovrebbe gia' essere
in grado di compilare le versioni OpenMP, MPI e/o CUDA eventualmente
presenti nella directory stessa. Si puo' modificare il Makefile
fornito, oppure si puo' decidere di non usarlo ed effettuare la
compilazione in modo manuale. In tal caso specificare in questo file i
comandi da usare per la compilazione dei programmi.

# HPC convex hull 2019/2020

Questo archivio contiene le implementazioni e la relazione per il progetto 
del corso di High Performance Computing, laurea in Ingegneria e Scienze
Informatiche, Universita' di Bologna sede di Cesena, AA 2019/2020.

## Compilazione

Ho modificato leggermente il Makefile fornito ed ho creato altri Makefile che
vengono invocati ricorsivamente per creare i file di input (quelli ricreabili),
generare gli output `*.hull` e le immagini generate da `plot-hull.gp`.

Per compilare i file presenti nella cartella `src`, spostarsi nella cartella e
usare il comando `make`:
```sh
path/> cd src
path/src> make
```

Per generare i file di input come presente nel Makefile originale, usare:
```sh
path/src> make files
```

TODO

