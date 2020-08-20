/****************************************************************************
 *
 * convex-hull.c
 *
 * Compute the convex hull of a set of points in 2D
 *
 * Copyright (C) 2019 Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last updated on 2019-11-25
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 *
 * Questo programma calcola l'inviluppo convesso (convex hull) di un
 * insieme di punti 2D letti da standard input usando l'algoritmo
 * "gift wrapping". Le coordinate dei vertici dell'inviluppo sono
 * stampate su standard output.  Per una descrizione completa del
 * problema si veda la specifica del progetto sul sito del corso:
 *
 * http://moreno.marzolla.name/teaching/HPC/
 *
 * Per compilare:
 *
 * gcc -D_XOPEN_SOURCE=600 -std=c99 -Wall -Wpedantic -O2 convex-hull.c -o convex-hull -lm
 *
 * (il flag -D_XOPEN_SOURCE=600 e' superfluo perche' viene settato
 * nell'header "hpc.h", ma definirlo tramite la riga di comando fa si'
 * che il programma compili correttamente anche se non si include
 * "hpc.h", o per errore non lo si include come primo file).
 *
 * Per eseguire il programma si puo' usare la riga di comando:
 *
 * ./convex-hull < ace.in > ace.hull
 * 
 * Per visualizzare graficamente i punti e l'inviluppo calcolato è
 * possibile usare lo script di gnuplot (http://www.gnuplot.info/)
 * incluso nella specifica del progetto:
 *
 * gnuplot -c plot-hull.gp ace.in ace.hull ace.png
 *
 ****************************************************************************/
#include <mpi.h>
#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* A single point */
typedef struct {
    double x, y;
} point_t;

typedef struct {
    point_t cur, next;
} reduce_point_t;

/* An array of n points */
typedef struct {
    int n;      /* number of points     */
    point_t *p; /* array of points      */
} points_t;

enum {
    LEFT = -1,
    COLLINEAR,
    RIGHT
};

enum {
    LEFTMOST = 0,
    HIGHEST = 1,
    RIGHTMOST = 2,
    LOWEST = 3
};

/**
 * Read input from file f, and store the set of points into the
 * structure pset.
 */
void read_input( FILE *f, points_t *pset )
{
    char buf[1024];
    int i, dim, npoints;
    point_t *points;
    
    if ( 1 != fscanf(f, "%d", &dim) ) {
        fprintf(stderr, "FATAL: can not read dimension\n");
        exit(EXIT_FAILURE);
    }
    if (dim != 2) {
        fprintf(stderr, "FATAL: This program supports dimension 2 only (got dimension %d instead)\n", dim);
        exit(EXIT_FAILURE);
    }
    if (NULL == fgets(buf, sizeof(buf), f)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: failed to read rest of first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != fscanf(f, "%d", &npoints)) {
        fprintf(stderr, "FATAL: can not read number of points\n");
        exit(EXIT_FAILURE);
    }
    assert(npoints > 2);
    points = (point_t*)malloc( npoints * sizeof(*points) );
    assert(points);
    for (i=0; i<npoints; i++) {
        if (2 != fscanf(f, "%lf %lf", &(points[i].x), &(points[i].y))) {
            fprintf(stderr, "FATAL: failed to get coordinates of point %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    pset->n = npoints;
    pset->p = points;
}

/**
 * Free the memory allocated by structure pset.
 */
void free_pointset( points_t *pset )
{
    pset->n = 0;
    free(pset->p);
    pset->p = NULL;
}

/**
 * Dump the convex hull to file f. The first line is the number of
 * dimensione (always 2); the second line is the number of vertices of
 * the hull PLUS ONE; the next (n+1) lines are the vertices of the
 * hull, in clockwise order. The first point is repeated twice, in
 * order to be able to plot the result using gnuplot as a closed
 * polygon
 */
void write_hull( FILE *f, const points_t *hull )
{
    int i;
    fprintf(f, "%d\n%d\n", 2, hull->n + 1);
    for (i=0; i<hull->n; i++) {
        fprintf(f, "%f %f\n", hull->p[i].x, hull->p[i].y);
    }
    /* write again the coordinates of the first point */
    fprintf(f, "%f %f\n", hull->p[0].x, hull->p[0].y);    
}

/** 
 * Compute the euclidean distance between two points. 
 */
double dist(const point_t a, const point_t b){
    return hypot(a.x - b.x, a.y - b.y);
}

/**
 * Return LEFT, RIGHT or COLLINEAR depending on the shape
 * of the vectors p0p1 and p1p2
 *
 * LEFT            RIGHT           COLLINEAR
 * 
 *  p2              p1----p2            p2
 *    \            /                   /
 *     \          /                   /
 *      p1       p0                  p1
 *     /                            /
 *    /                            /
 *  p0                            p0
 *
 * See Cormen, Leiserson, Rivest and Stein, "Introduction to Algorithms",
 * 3rd ed., MIT Press, 2009, Section 33.1 "Line-Segment properties"
 */
int turn(const point_t p0, const point_t p1, const point_t p2)
{
    /*
      This function returns the correct result (COLLINEAR) also in the
      following cases:
      
      - p0==p1==p2
      - p0==p1
      - p1==p2
    */
    const double cross = (p1.x-p0.x)*(p2.y-p0.y) - (p2.x-p0.x)*(p1.y-p0.y);
    if (cross > 0.0) {
        return LEFT;
    } else {
        if (cross < 0.0) {
            return RIGHT;
        } else {
            return COLLINEAR;
        }
    }
}

int check_turn_left(const point_t p0, const point_t p1, const point_t p2) {
    int turning = turn(p0, p1, p2);
    if (turning == LEFT ||
        (turning == COLLINEAR && dist(p0, p2) > dist(p0, p1))) {
        return 1;
    }
    return 0;
}

void turn_reduce(void *in, void *inout, int *len, MPI_Datatype *dptr) {
    int i;

    reduce_point_t *in_conv = (reduce_point_t*)in;
    reduce_point_t *inout_conv = (reduce_point_t*)inout;

    for (i=0; i<*len; i++) {
        reduce_point_t out = in_conv[i];
        if (check_turn_left(out.cur, inout_conv[i].next, out.next)) {
            inout_conv[i] = out;
        }
    }
}

/**
 * Get the clockwise angle between the line p0p1 and the vector p1p2 
 *
 *         .
 *        . 
 *       .--+ (this angle) 
 *      .   |    
 *     .    V
 *    p1--------------p2
 *    /
 *   /
 *  /
 * p0
 *
 * The function is not used in this program, but it might be useful.
 */
double cw_angle(const point_t p0, const point_t p1, const point_t p2)
{
    const double x1 = p2.x - p1.x;
    const double y1 = p2.y - p1.y;    
    const double x2 = p1.x - p0.x;
    const double y2 = p1.y - p0.y;
    const double dot = x1*x2 + y1*y2;
    const double det = x1*y2 - y1*x2;
    const double result = atan2(det, dot);
    return (result >= 0 ? result : 2*M_PI + result);
}

void divide_set(const points_t *pset, const int p1_index, const int p2_index, points_t *res_set) {
    int n = pset->n;
    point_t *p = pset->p;
    int i;

    res_set->n = 1;
    res_set->p = (point_t*)malloc(n * sizeof(point_t)); assert(res_set->p);
    res_set->p[0] = p[p1_index];

    for (i = 0; i < n; i++) {
        if (i == p1_index || i == p2_index) continue;
        if (turn(p[p1_index], p[p2_index], p[i]) == LEFT) {
            res_set->p[res_set->n++] = p[i];
        }
    }

    res_set->p[res_set->n++] = p[p2_index];
    res_set->p = (point_t*)realloc(res_set->p, res_set->n * sizeof(point_t)); assert(res_set->p);
}

/**
 * Compute the convex hull of all points in pset using the "Gift
 * Wrapping" algorithm. The vertices are stored in the hull data
 * structure, that does not need to be initialized by the caller.
 */
void partial_convex_hull(const points_t *pset, points_t *hull, int startIndex, int endIndex, int rank, int n_procs)
{
    int n, i, j;
    point_t *p;

    MPI_Datatype mpi_point_t;
    MPI_Type_contiguous(2, MPI_DOUBLE, &mpi_point_t);
    MPI_Type_commit(&mpi_point_t);

    MPI_Datatype mpi_reduce_point_t;
    MPI_Type_contiguous(2, mpi_point_t, &mpi_reduce_point_t);
    MPI_Type_commit(&mpi_reduce_point_t);

    MPI_Op mpi_turn_reduce;
    MPI_Op_create(turn_reduce, 1, &mpi_turn_reduce);

    if (rank == 0) {
        n = pset->n;
        p = pset->p;

        hull->n = 0;
        /* There can be at most n points in the convex hull. At the end of
        this function we trim the excess space. */
        hull->p = (point_t*)malloc(n * sizeof(*(hull->p))); assert(hull->p);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / n_procs;

    int *sendcounts = (int*)malloc(n_procs * sizeof(int));
    int *displs = (int*)malloc(n_procs * sizeof(int));

    int cnt = 0;
    for (i=0; i<n_procs; i++) {
        int cntnow = n / n_procs + ((i < n%n_procs) ? 1 : 0);
        sendcounts[i] = cntnow;
        displs[i] = cnt;
        cnt += cntnow;

        if (sendcounts[i] == 0) {
            sendcounts[i] = 1;
            displs[i] = 0;
        }

        printf("Proc %d send %d displ %d\n", rank, sendcounts[i], displs[i]);
    }

    point_t local_cur, local_next, local_end;
    point_t *local_p = (point_t*)malloc((local_n+1) * sizeof(point_t));

    if (rank == 0) {
        local_cur = p[startIndex];
        local_end = p[endIndex];
    }
    
    MPI_Bcast(&local_cur, 1, mpi_point_t, 0, MPI_COMM_WORLD);
    MPI_Bcast(&local_end, 1, mpi_point_t, 0, MPI_COMM_WORLD);

    MPI_Scatterv(p, sendcounts, displs, mpi_point_t, local_p, local_n+1, mpi_point_t, 0, MPI_COMM_WORLD);
 
    /* Main loop of the Gift Wrapping algorithm. This is where most of
       the time is spent; therefore, this is the block of code that
       must be parallelized. */
    do {
        if (rank == 0) {
            /* Add the current vertex to the hull */
            assert(hull->n < n);
            hull->p[hull->n] = local_cur;
            hull->n++;
        }

        local_next = local_p[0];
        for (j=1; j<sendcounts[rank]; j++) {
            /* Check if segment turns left */
            if (check_turn_left(local_cur, local_next, local_p[j])) {
                local_next = local_p[j];
            }
        }

        reduce_point_t cur_and_next = {local_cur, local_next};
        reduce_point_t final_cur_and_next;

        MPI_Allreduce(&cur_and_next, &final_cur_and_next, 1, mpi_reduce_point_t, mpi_turn_reduce, MPI_COMM_WORLD);

        local_cur = final_cur_and_next.next;
    } while (!(local_cur.x == local_end.x && local_cur.y == local_end.y));

    free(local_p);

    // if (rank == 0) {
    //     /* Trim the excess space in the convex hull array */
    //     hull->p = (point_t*)realloc(hull->p, (hull->n) * sizeof(*(hull->p)));
    //     assert(hull->p); 
    // }
}

void convex_hull(const points_t *pset, points_t *hull, int rank, int n_procs)
{
    int n, i, j, next, n_hull = 0;
    point_t *p;

    points_t partial_sets[4];

    if (rank == 0) {
        n = pset->n;
        p = pset->p;

        /* Identify the 4 cardinal points in the set. */
        int cardinal[] = {0, 0, 0, 0};
        for (i = 1; i < n; i++) {
            /* Leftmost-down */
            if (p[i].x < p[cardinal[LEFTMOST]].x || (p[i].x == p[cardinal[LEFTMOST]].x && p[i].y < p[cardinal[LEFTMOST]].y)) {
                cardinal[LEFTMOST] = i;
            }
            /* Rightmost-up */
            if (p[i].x > p[cardinal[RIGHTMOST]].x || (p[i].x == p[cardinal[RIGHTMOST]].x && p[i].y > p[cardinal[RIGHTMOST]].y)) {
                cardinal[RIGHTMOST] = i;
            } 
            /* Highest-left */
            if (p[i].y > p[cardinal[HIGHEST]].y || (p[i].y == p[cardinal[HIGHEST]].y && p[i].x < p[cardinal[HIGHEST]].x)) {
                cardinal[HIGHEST] = i;
            } 
            /* Lowest-right */
            if (p[i].y < p[cardinal[LOWEST]].y || (p[i].y == p[cardinal[LOWEST]].y && p[i].x > p[cardinal[LOWEST]].x)) {
                cardinal[LOWEST] = i;
            }
        }

        /* Divide the plane in 4 parts */
        divide_set(pset, cardinal[LEFTMOST], cardinal[HIGHEST], &partial_sets[LEFTMOST]);
        divide_set(pset, cardinal[HIGHEST], cardinal[RIGHTMOST], &partial_sets[HIGHEST]);
        divide_set(pset, cardinal[RIGHTMOST], cardinal[LOWEST], &partial_sets[RIGHTMOST]);
        divide_set(pset, cardinal[LOWEST], cardinal[LEFTMOST], &partial_sets[LOWEST]);
    }

    /* Calculate every partial hull */
    points_t partial_hulls[4];
    for (j = 0; j < 4; j++) {
        partial_convex_hull(&partial_sets[j], &partial_hulls[j], 0, partial_sets[j].n - 1, rank, n_procs);
    }

    /* Merge hulls */
    if (rank == 0) {
        for (j = 0; j < 4; j++) {
            n_hull += partial_hulls[j].n;
        }
        hull->n = n_hull;
        hull->p = (point_t*)malloc(n_hull * sizeof(point_t)); assert(hull->p);

        next = 0;
        for (j = 0; j < 4; j++) {
            for (i = 0; i < partial_hulls[j].n; i++) {
                hull->p[next++] = partial_hulls[j].p[i];
            }
            free_pointset(&partial_sets[j]);
            free_pointset(&partial_hulls[j]);
        }
    }
}

/**
 * Compute the area ("volume", in qconvex terminoloty) of a convex
 * polygon whose vertices are stored in pset using Gauss' area formula
 * (also known as the "shoelace formula"). See:
 *
 * https://en.wikipedia.org/wiki/Shoelace_formula
 *
 * This function does not need to be parallelized.
 */
double hull_volume( const points_t *hull )
{
    const int n = hull->n;
    const point_t *p = hull->p;
    double sum = 0.0;
    int i;
    for (i=0; i<n-1; i++) {
        sum += ( p[i].x * p[i+1].y - p[i+1].x * p[i].y );
    }
    sum += p[n-1].x*p[0].y - p[0].x*p[n-1].y;
    return 0.5*fabs(sum);
}

/**
 * Compute the length of the perimeter ("facet area", in qconvex
 * terminoloty) of a convex polygon whose vertices are stored in pset.
 * This function does not need to be parallelized.
 */
double hull_facet_area( const points_t *hull )
{
    const int n = hull->n;
    const point_t *p = hull->p;
    double length = 0.0;
    int i;
    for (i=0; i<n-1; i++) {
        length += hypot( p[i].x - p[i+1].x, p[i].y - p[i+1].y );
    }
    /* Add the n-th side connecting point n-1 to point 0 */
    length += hypot( p[n-1].x - p[0].x, p[n-1].y - p[0].y );
    return length;
}

int main( int argc, char *argv[]  )
{
    points_t pset, hull;
    double tstart, elapsed;

    int rank, n_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    
    if (rank == 0){
        read_input(stdin, &pset);
    }

    tstart = hpc_gettime();
    convex_hull(&pset, &hull, rank, n_procs);
    elapsed = hpc_gettime() - tstart;

    if (rank == 0) {
        fprintf(stderr, "\nConvex hull of %d points in 2-d:\n\n", pset.n);
        fprintf(stderr, "MPI computation using %d processors\n", n_procs);
        fprintf(stderr, "  Number of vertices: %d\n", hull.n);
        fprintf(stderr, "  Total facet area: %f\n", hull_facet_area(&hull));
        fprintf(stderr, "  Total volume: %f\n\n", hull_volume(&hull));
        fprintf(stderr, "Elapsed time: %f\n\n", elapsed);
        write_hull(stdout, &hull);
        free_pointset(&pset);
        free_pointset(&hull);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;    
}
