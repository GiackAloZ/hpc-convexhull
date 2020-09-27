/*****************************************************
 * 
 * Giacomo Aloisi (giacomo.aloisi@studio.unibo.it)
 * Matr. 0000832933
 * 
 * Versione OpenMP no split
 * 
******************************************************/

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

/* A point for the omp reduction */
typedef struct {
    point_t *set;
    int index;
    int cur_index;
} red_point_t;

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
 * User-defined reduction function for the omp reduction clause.
 * Each structure has the same `set` and `cur_index` point.
 * The function compares the points given with `check_next_chpoint`.
 */
red_point_t check_next_chpoint_red(red_point_t p1, red_point_t p2) {
    if (check_next_chpoint(p1.set[p1.cur_index], p1.set[p1.index], p2.set[p2.index]))
        return p2;
    return p1;
}

/**
 * Compute the convex hull of all points in pset using a parallelized
 * version of the "Gift  Wrapping" algorithm.
 * The vertices are stored in the hull data
 * structure, that does not need to be initialized by the caller.
 */
void convex_hull(const points_t *pset, points_t *hull)
{
    int n = pset->n;
    point_t *p = pset->p;
    int i, cur, next, leftmost;
    
    /* Initalize the hull structure. */
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
    cur = leftmost;

#pragma omp declare reduction (left_turn_red:red_point_t:omp_out = check_next_chpoint_red(omp_out, omp_in)) initializer(omp_priv = omp_orig)
 
    /* Outer loop of the Gift Wrapping algorithm. */
    do {
        /* Add the current vertex to the hull */
        assert(hull->n < n);
        hull->p[hull->n] = p[cur];
        hull->n++;
        
        /* Search for the next vertex */
        /* Initialize reduction result and next index */
        next = (cur + 1) % n;
        red_point_t res = {p, next, cur};

        /* Inner loop. */
#pragma omp parallel for reduction(left_turn_red:res) default(none) shared(n) shared(cur) shared(p) private(i)
        for (i=0; i<n; i++) {
            /* Check if point i is candidate for next vertex. */
            if (check_next_chpoint(p[cur], p[res.index], p[i])) {
                res.index = i;
            }
        }

        /* Extract reduction result and assign to next. */
        next = res.index;
        assert(cur != next);
        cur = next;
    } while (cur != leftmost); /* Stop computation when returned to the start. */
    
    /* Trim the excess space in the convex hull array */
    hull->p = (point_t*)realloc(hull->p, (hull->n) * sizeof(*(hull->p)));
    assert(hull->p); 
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

int main( void )
{
    points_t pset, hull;
    double tstart, elapsed;
    
    read_input(stdin, &pset);
    tstart = hpc_gettime();
    convex_hull(&pset, &hull);
    elapsed = hpc_gettime() - tstart;
    fprintf(stderr, "\nConvex hull of %d points in 2-d:\n\n", pset.n);
    fprintf(stderr, "OMP computation using %d threads\n", omp_get_max_threads());
    fprintf(stderr, "  Number of vertices: %d\n", hull.n);
    fprintf(stderr, "  Total facet area: %f\n", hull_facet_area(&hull));
    fprintf(stderr, "  Total volume: %f\n\n", hull_volume(&hull));
    fprintf(stderr, "Elapsed time: %f\n\n", elapsed);
    write_hull(stdout, &hull);
    free_pointset(&pset);
    free_pointset(&hull);
    return EXIT_SUCCESS;    
}
