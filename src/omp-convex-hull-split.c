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

/* Compute the euclidean distance between two points (squared). */
long long int square_dist(const point_t a, const point_t b){
    long long int x = a.x - b.x;
    long long int y = a.y - b.y;
    return x*x + y*y;
}

/* Compute a partial hull with the Gift Wrapping algorithm,
considering two points as start and end of the partial hull. */
void partial_convex_hull(const points_t *pset, points_t *hull, int startIndex, int endIndex){
    const int n = pset->n;
    const point_t *p = pset->p;
    int i, j;
    int cur, next;
    
    /* Get number of threads to use */
    int n_threads = omp_get_max_threads();
    /* Array for partial computation of next */
    int* next_priv = (int*)malloc(n_threads * sizeof(int));
    
    hull->n = 0;
    /* There can be at most n points in the convex hull. At the end of
       this function we trim the excess space. */
    hull->p = (point_t*)malloc(n * sizeof(*(hull->p))); assert(hull->p);

    cur = startIndex;
 
   /* 
        Main loop of the Gift Wrapping algorithm. This is where most of
        the time is spent; therefore, this is the block of code that
        must be parallelized. 
        
        A batch of threads is created before the main loop and kept through the duration of it.
    */
#pragma omp parallel default(none) firstprivate(n) private(i) shared(n_threads) shared(leftmost) shared(hull) shared(cur) shared(p) shared(next_priv) shared(next)
    {
        int tid = omp_get_thread_num();

        do {
#pragma omp single
            {
                /* Add the current vertex to the hull */
                assert(hull->n < n);
                hull->p[hull->n] = p[cur];
                hull->n++;
            }
            
            /* Search for the next vertex */
            /* Initialize next_priv for each thread as the next point in the set */
            next_priv[tid] = (cur + 1) % n;
#pragma omp barrier

            /* The actual parallel-heavy computation */
#pragma omp for private(j)
            for (j=0; j<n; j++) {
                /* Check if segment turns left */
                if (LEFT == turn(p[cur], p[next_priv[tid]], p[j])) {
                    next_priv[tid] = j;
                }
            }

            /* Sequential reduction */
#pragma omp single
            {
                next = next_priv[0];
                for (i = 1; i < n_threads; i++){
                    if (LEFT == turn(p[cur], p[next], p[next_priv[i]])){
                        next = next_priv[i];
                    }
                }

                assert(cur != next);
                cur = next;
            }
        } while (cur != endIndex);
    }
    
    /* Trim the excess space in the convex hull array */
    hull->p = (point_t*)realloc(hull->p, (hull->n) * sizeof(*(hull->p)));
    assert(hull->p);
    free(next_priv);
}

/**
 * Compute the convex hull of all points in pset using the "Gift
 * Wrapping" algorithm. The vertices are stored in the hull data
 * structure, that does not need to be initialized by the caller.
 */
void convex_hull(const points_t *pset, points_t *hull)
{
    const int n = pset->n;
    const point_t *p = pset->p;
    int i;
    int next, leftmost, rightmost;
    
    hull->n = 0;
    /* There can be at most n points in the convex hull. At the end of
       this function we trim the excess space. */
    hull->p = (point_t*)malloc(n * sizeof(*(hull->p))); assert(hull->p);
    
    /* Identify the leftmost-lower point p[leftmost] and rightmost-upper point p[rightmost]*/
    leftmost = 0;
    rightmost = 0;
    for (i = 1; i<n; i++) {
        if (p[i].x < p[leftmost].x || (p[i].x == p[leftmost].x && p[i].y < p[leftmost].y)) {
            leftmost = i;
        }
        if (p[i].x > p[rightmost].x || (p[i].x == p[rightmost].x && p[i].y > p[rightmost].y)) {
            rightmost = i;
        }
    }    

    /* Divide the plane in half, taking points under or above the line between leftmost and rightmost points */
    points_t upper_set, lower_set;
    upper_set.p = (point_t*)malloc(n * sizeof(point_t)); assert(upper_set.p);
    lower_set.p = (point_t*)malloc(n * sizeof(point_t)); assert(lower_set.p);

    upper_set.p[0] = p[leftmost];
    upper_set.n = 1;
    lower_set.p[0] = p[rightmost];
    lower_set.n = 1;

    for (i = 0; i < n; i++) {
        /* Special case */
        if (i == leftmost || i == rightmost)
            continue;
        /* Point in upper set (turning LEFT so ABOVE the line) */
        if (turn(p[leftmost], p[rightmost], p[i]) == LEFT) {
            upper_set.p[upper_set.n++] = p[i];
        } /* Point in lower set (turning RIGHT so UNDER the line) */
        else if (turn(p[leftmost], p[rightmost], p[i]) == RIGHT) {
            lower_set.p[lower_set.n++] = p[i];
        }
    }
    
    upper_set.p[upper_set.n++] = p[rightmost];
    lower_set.p[lower_set.n++] = p[leftmost];
    
    /* Realloc partial sets with appropriate sizes. */ 
    upper_set.p = (point_t*)realloc(upper_set.p, upper_set.n * sizeof(point_t)); assert(upper_set.p);
    lower_set.p = (point_t*)realloc(lower_set.p, lower_set.n * sizeof(point_t)); assert(lower_set.p);

    /* Calculate upper and lower hulls.*/
    points_t lower_hull, upper_hull;
    partial_convex_hull(&upper_set, &upper_hull, 0, upper_set.n - 1);
    partial_convex_hull(&lower_set, &lower_hull, 0, lower_set.n - 1);
    
    /* Merge hulls. */
    hull->n = upper_hull.n + lower_hull.n;
    hull->p = (point_t*)realloc(hull->p, hull->n * sizeof(point_t)); assert(hull->p);

    next = 0;
    for (i = 0; i < upper_hull.n; i++) {
        hull->p[next++] = upper_hull.p[i];
    }
    for (i = 0; i < lower_hull.n; i++) {
        hull->p[next++] = lower_hull.p[i];
    }

    /* Free partial hulls. */
    free_pointset(&upper_hull);
    free_pointset(&lower_hull);
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
