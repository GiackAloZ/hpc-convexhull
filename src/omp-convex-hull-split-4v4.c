#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

/* A single point */
typedef struct {
    double x, y;
} point_t;

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
 * Compute the euclidean distance between two points. 
 */
double dist(const point_t a, const point_t b){
    return hypot(a.x - b.x, a.y - b.y);
}

/* Enumeration for indexes of partial sets. */
enum {
    LEFTMOST = 0,
    HIGHEST = 1,
    RIGHTMOST = 2,
    LOWEST = 3
};

/** 
 * Divide a set of points in another set, considering points
 * that turn left according to the line p1--->p2.
 * 
 * So, every point p that is like the figure:
 * 
 *   p                     p2
 *    \                    /\
 *     \                  /  \
 *      p2      OR       p    \
 *     /                       \
 *    /                         \
 *  p1                           p1  
 * 
 * Will be placed in the res_set.
 * The points p1 and p2 are stored at the beginning and at the end of the res_set. 
 */
void divide_set(const points_t *pset, const int p1_index, const int p2_index, points_t *res_set) {
    int n = pset->n;
    point_t *p = pset->p;
    int i;

    /* Set up res_set. */
    res_set->n = 1;
    res_set->p = (point_t*)malloc(n * sizeof(point_t)); assert(res_set->p);
    res_set->p[0] = p[p1_index];

    for (i = 0; i < n; i++) {
        /* Ignore points used for line. */
        if (i == p1_index || i == p2_index) continue;
        if (turn(p[p1_index], p[p2_index], p[i]) == LEFT) {
            res_set->p[res_set->n++] = p[i];
        }
    }

    res_set->p[res_set->n++] = p[p2_index];
    /* Resize res_set. */
    res_set->p = (point_t*)realloc(res_set->p, res_set->n * sizeof(point_t)); assert(res_set->p);
}

/** 
 * Compute a partial hull with the Gift Wrapping algorithm,
 * considering two points as start and end of the partial hull. 
 * 
 * The inner loop of the Gift Wrapping algo is parallelized
 * with a REDUCTION on the points using the TURN function using n_threads threads.
 */
void partial_convex_hull(const points_t *pset, points_t *hull, int startIndex, int endIndex, int n_threads){
    const int n = pset->n;
    const point_t *p = pset->p;
    int i, j;
    int cur, next;

    /* Array for partial computation of next */
    int* next_priv = (int*)malloc(n_threads * sizeof(int));
    
    hull->n = 0;
    /* There can be at most n points in the convex hull. At the end of
       this function we trim the excess space. */
    hull->p = (point_t*)malloc(n * sizeof(*(hull->p))); assert(hull->p);

    cur = startIndex;
 
    /* Main loop of the Gift Wrapping algorithm. This is where most of
       the time is spent; therefore, this is the block of code that
       must be parallelized. */
    do {
        /* Add the current vertex to the hull */
        assert(hull->n < n);
        hull->p[hull->n] = p[cur];
        hull->n++;
        
        /* Search for the next vertex */
        /* Initialize next_priv for each thread as the next point in the set:
        it will be actually any other point that is not cur */
        for(i = 0; i < n_threads; i++)
            next_priv[i] = (cur + 1) % n;   /* Modulo is added to prevent access out of memory */

        /* Parallelizing inner loop with a manual reduction */
#pragma omp parallel for default(none) private(j) shared(next_priv) shared(cur) shared(p) num_threads(n_threads)
        for (j=0; j<n; j++) {
            int tid = omp_get_thread_num();
            int turning = turn(p[cur], p[next_priv[tid]], p[j]);
            /* Check if segment turns left */
            if (turning == LEFT ||  /* If collinear, take the furthers point from cur */
                (turning == COLLINEAR && dist(p[cur], p[j]) > dist(p[cur], p[next_priv[tid]]))){
                /* Update private next point for the current thread */
                next_priv[tid] = j;
            }
        }

        /* Reduce all next_priv into one single next */
        next = next_priv[0];
        for (i = 1; i < n_threads; i++){
            int turning = turn(p[cur], p[next], p[next_priv[i]]);
            if (turning == LEFT ||
                (turning == COLLINEAR && dist(p[cur], p[next_priv[i]]) > dist(p[cur], p[next]))){
                next = next_priv[i];
            }
        }

        assert(cur != next);
        cur = next;
    } while (cur != endIndex);
    
    /* Trim the excess space in the convex hull array */
    hull->p = (point_t*)realloc(hull->p, (hull->n) * sizeof(*(hull->p)));
    assert(hull->p);
    free(next_priv);
}

/**
 * Compute the convex hull of all points in pset using the "Gift
 * Wrapping" algorithm. The vertices are stored in the hull data
 * structure, that does not need to be initialized by the caller.
 * 
 * The algorithm divides the original set of points into four sets:
 *      - taking 4 "cardinal" points (highest, lowest, rightmos, leftmost) of the set, we divide the sets
 *        tracing lines from leftmost to highest, from highest to righmost (we take points above these lines),
 *        from rightmost to lowest, from lowest to leftmost (we take points below these lines)
 *      - each hull is computed for every set in parallel
 *      - these 4 hulls are than concatenated into one final hull
 * 
 * The reduced sets allow the computation to be faster because the algorithm needs to iterate on
 * a smaller number of points.
 * Note that the 4 "cardinal" points are always part of the convex hull of pset.
 */
void convex_hull(const points_t *pset, points_t *hull)
{
    const int n = pset->n;
    const point_t *p = pset->p;
    int i, j, n_threads;
    
    /* Identify the 4 "cardinal" points in the set. */
    int cardinal[] = {0, 0, 0, 0};
    for (i = 1; i < n; i++) {
        /* Break ties by taking the other coordinate and check for biggest/lowest values too. */

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

    /* Initialize final hull structure. */
    hull->n = 0;
    hull->p = (point_t*)malloc(n * sizeof(point_t)); assert(hull->p);

    /* Prepare partial hulls and split threads to sets. */
    points_t partial_hulls[4], partial_sets[4];
    n_threads = min(omp_get_max_threads(), 4);

/* Divide the original set into partial ones splitted
   accordingly to the line of two cardinal points. */
#pragma omp parallel for default(none) private(j) \
        shared(pset) shared(partial_sets) shared(cardinal) \
        num_threads(n_threads)
    for (j = 0; j < 4; j++) {
            divide_set(pset, cardinal[j], cardinal[(j+1) % 4], &partial_sets[j]);
    }

    int partial_sets_sum = 0;
    for (j = 0; j < 4; j++) {
        partial_sets_sum += partial_sets[j].n;
    }


/* Main loop that does the computation in parallel for each slice of the original set. */
#pragma omp parallel for default(none) private(j) \
        shared(pset) shared(partial_hulls) shared(partial_sets) shared(partial_sets_sum) shared(cardinal) shared(stderr) \
        num_threads(n_threads)
    for (j = 0; j < 4; j++) {
        int priv_nthreads = max(1, round((((double)partial_sets[j].n) / partial_sets_sum) * omp_get_max_threads()));
        // fprintf(stderr, "\nStart ch with %d threads\n\n", priv_nthreads);
        // fprintf(stderr, "\nRatio %f\n\n", ((double)partial_sets[j].n) / partial_sets_sum);

        /* Compute the convex hull of the partial set. */
        partial_convex_hull(&partial_sets[j], &partial_hulls[j], 0, partial_sets[j].n - 1, priv_nthreads);

        free_pointset(&partial_sets[j]);
    }

    /* Add the obtained hulls to the final one. */
    for (j = 0; j < 4; j++) {
        for (i = 0; i < partial_hulls[j].n; i++) {
            hull->p[hull->n++] = partial_hulls[j].p[i];
        }
        free_pointset(&partial_hulls[j]);
    }

    /* Resize the hull with its actual size. */
    hull->p = (point_t*)realloc(hull->p, hull->n * sizeof(point_t)); assert(hull->p);
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
