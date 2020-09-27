/*****************************************************
 * 
 * Giacomo Aloisi (giacomo.aloisi@studio.unibo.it)
 * Matr. 0000832933
 * 
 * Versione MPI no split
 * 
******************************************************/

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

/* A point for the MPI reduction */
typedef struct {
    point_t cur, next;
} reduce_point_t;

/* An array of n points */
typedef struct {
    int n;      /* number of points     */
    point_t *p; /* array of points      */
} points_t;

/* Turn results */
enum {
    LEFT = -1,
    COLLINEAR,
    RIGHT
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
    if (pset->n > 0) {
        free(pset->p);
    }

    pset->n = 0;
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
 * Computes the squared euclidean distance between two points.
 * Used to check for the furthest of three collinear points.
 */
double square_dist(const point_t a, const point_t b){
    const double x = a.x - b.x;
    const double y = a.y - b.y;
    return x*x + y*y;
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

/**
 * Checks if the point `tocheck` is the next point of the convex hull given the `cur` point and the current `next` possible convex hull vertex.
 * This function checks for 2 things:
 * - the line cur->next->tocheck turns left
 * - the points are collinear AND `tocheck` is further from `cur` than `next` is from `cur`
 * 
 * If one of the two conditions is true, then `tocheck` replaces `next` has the next possible vertex in the convex hull computation.
 * This function is used to find the minimal convex hull.
 */
int check_next_chpoint(const point_t cur, const point_t next, const point_t tocheck) {
    int turning = turn(cur, next, tocheck);
    if (turning == LEFT ||
        (turning == COLLINEAR && square_dist(cur, tocheck) > square_dist(cur, next))) {
        return 1;
    }
    return 0;
}

/**
 * User-defined reduction function for the MPI reduction clause.
 * 
 * The in and out parameters are pointers to the reduece_point_t struct that
 * stores the information the cur vertex and the next point (actual point represented)
 * as the result of the reduction.
 * 
 * The function needs to handle multiple input length vectors, specified
 * by the parameter `len`.
 * 
 * The output of the reduction must be stored in the same memory address
 * as the `inout` pointer.
 * 
 * The function compares the points given with `check_next_chpoint`.
 */
void turn_reduce(void *in, void *inout, int *len, MPI_Datatype *dptr) {
    int i;

    reduce_point_t *in_conv = (reduce_point_t*)in;
    reduce_point_t *inout_conv = (reduce_point_t*)inout;

    for (i=0; i<*len; i++) {
        reduce_point_t out = in_conv[i];
        if (check_next_chpoint(out.cur, inout_conv[i].next, out.next)) {
            inout_conv[i] = out;
        }
    }
}

/**
 * Compute the convex hull of all points in pset using a parallelized
 * version of the "Gift  Wrapping" algorithm.
 * The vertices are stored in the hull data
 * structure, that does not need to be initialized by the caller.
 */
void convex_hull(const points_t *pset, points_t *hull, int rank, int n_procs)
{
    int n, i;
    point_t *p = pset->p;
    int leftmost;

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

        /* Allocate hull structure. */
        hull->n = 0;
        /* There can be at most n points in the convex hull. At the end of
        this function we trim the excess space. */
        hull->p = (point_t*)malloc(n * sizeof(*(hull->p))); assert(hull->p);

        /* Identify the leftmost point p[leftmost] */
        leftmost = 0;
        for (i = 1; i<n; i++) {
            if (p[i].x < p[leftmost].x) {
                leftmost = i;
            }
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Calculate count and displacements for Scatterv. */
    int local_n = n / n_procs;
    int *sendcounts = (int*)malloc(n_procs * sizeof(int));
    int *displs = (int*)malloc(n_procs * sizeof(int));

    int cnt = 0;
    for (i=0; i<n_procs; i++) {
        /* Fix local count if n is not divisible by nprocs. */
        int cntnow = local_n + ((i < n%n_procs) ? 1 : 0);
        sendcounts[i] = cntnow;
        displs[i] = cnt;
        cnt += cntnow;

        /* Fix sendcount zero, just send the first point again */
        if (sendcounts[i] == 0) {
            sendcounts[i] = 1;
            displs[i] = 0;
        }
    }

    point_t local_cur, local_next;
    point_t *local_p = (point_t*)malloc((local_n+1) * sizeof(point_t));

    if (rank == 0) {
        local_cur = p[leftmost];
    }
    
    /* Broadcast start point of the convex hull segment. */
    MPI_Bcast(&local_cur, 1, mpi_point_t, 0, MPI_COMM_WORLD);
    /* Send local sets to each processor. */
    MPI_Scatterv(p, sendcounts, displs, mpi_point_t, local_p, local_n+1, mpi_point_t, 0, MPI_COMM_WORLD);

    point_t local_leftmost = local_cur;
 
    /* Outer loop of the Gift Wrapping algorithm. */
    do {
        if (rank == 0) {
            /* Add the current vertex to the hull */
            assert(hull->n < n);
            hull->p[hull->n] = local_cur;
            hull->n++;
        }

        /* Set local next as the first point of the local set. */
        local_next = local_p[0];
        /* Reduce local set of points which size is sendcounts[rank]. */
        for (i=1; i<sendcounts[rank]; i++) {
            /* Check if point i is candidate for next vertex. */
            if (check_next_chpoint(local_cur, local_next, local_p[i])) {
                local_next = local_p[i];
            }
        }

        /* Prepare reduction struct with local result. */
        reduce_point_t cur_and_next = {local_cur, local_next};
        reduce_point_t final_cur_and_next;

        /* Reduce all partial results. */
        MPI_Allreduce(&cur_and_next, &final_cur_and_next, 1, mpi_reduce_point_t, mpi_turn_reduce, MPI_COMM_WORLD);

        /* Store reduction result. */
        local_cur = final_cur_and_next.next;

        /* Stop computation when return to start point. */        
    } while (!(local_cur.x == local_leftmost.x && local_cur.y == local_leftmost.y));

    /* Free local set of points. */
    free(local_p);

    if (rank == 0) {
        /* Trim the excess space in the convex hull array */
        hull->p = (point_t*)realloc(hull->p, (hull->n) * sizeof(*(hull->p)));
        assert(hull->p); 
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
