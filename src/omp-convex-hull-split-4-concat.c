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

typedef struct {
    point_t *set;
    int index;
    int cur_index;
} red_point_t;

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
 * Computes the squared euclidean distance between two points.
 */
long long int square_dist(const point_t a, const point_t b){
    long long int x = a.x - b.x;
    long long int y = a.y - b.y;
    return x*x + y*y;
}

/**
 * Checks if the point `tocheck` is the next point of the convex hull given the `cur` point and the current `next` point.
 * This function checks for 2 things:
 * - the line cur->next->tocheck turns left
 * - the points are collinear AND `tocheck` is further from `cur` than `next` is from `cur`
 * 
 * If one of the two conditions is true, then `tocheck` replaces `next` in the convex hull computation.
 * This function is used to find the smallest convex hull set of points.
 */
int check_next_chpoint(const point_t cur, const point_t next, const point_t tocheck) {
    int turning = turn(cur, next, tocheck);
    if (turning == LEFT ||
        (turning == COLLINEAR && square_dist(cur, tocheck) > square_dist(cur, next))) {
        return 1;
    }
    return 0;
}

red_point_t check_next_chpoint_red(red_point_t p1, red_point_t p2) {
    if (check_next_chpoint(p1.set[p1.cur_index], p1.set[p1.index], p2.set[p2.index]))
        return p2;
    return p1;
}

/* Compute a partial hull with the Gift Wrapping algorithm,
considering two points as start and end of the partial hull. */
void partial_convex_hull(const points_t *pset, points_t *hull, int startIndex, int endIndex){
    const int n = pset->n;
    point_t *p = pset->p;
    int cur, next;
    
    hull->n = 0;
    /* There can be at most n points in the convex hull. At the end of
       this function we trim the excess space. */
    hull->p = (point_t*)malloc(n * sizeof(*(hull->p))); assert(hull->p);

    cur = startIndex;
 
#pragma omp declare reduction (left_turn_red:red_point_t:omp_out = check_next_chpoint_red(omp_out, omp_in)) initializer(omp_priv = omp_orig)
 
    /* Main loop of the Gift Wrapping algorithm. This is where most of
       the time is spent; therefore, this is the block of code that
       must be parallelized. */
    do {
        /* Add the current vertex to the hull */
        assert(hull->n < n);
        hull->p[hull->n] = p[cur];
        hull->n++;
        
        /* Search for the next vertex */
        /* Initialize reductionr result and next index */
        next = (cur + 1) % n;
        red_point_t res = {p, next, cur};

#pragma omp parallel for reduction(left_turn_red:res) default(none) firstprivate(next) firstprivate(n) firstprivate(cur) shared(p)
        for (int j=0; j<n; j++) {
            /* Check if segment turns left */
            if (check_next_chpoint(p[cur], p[res.index], p[j])) {
                res.index = j;
            }
        }

        next = res.index;
        assert(cur != next);
        cur = next;
    } while (cur != endIndex);
    
    /* Trim the excess space in the convex hull array */
    hull->p = (point_t*)realloc(hull->p, (hull->n) * sizeof(*(hull->p)));
    assert(hull->p);
}

enum {
    LEFTMOST = 0,
    HIGHEST = 1,
    RIGHTMOST = 2,
    LOWEST = 3
};

void divide_set(const points_t *pset, const int p1_index, const int p2_index, points_t *res_set) {
    int n = pset->n;
    point_t *p = pset->p;
    int i;

    res_set->n = 1;
    res_set->p = (point_t*)malloc(n * sizeof(point_t)); assert(res_set->p);
    res_set->p[0] = p[p1_index];

    if (p1_index == p2_index) {
        res_set->p = (point_t*)realloc(res_set->p, res_set->n * sizeof(point_t)); assert(res_set->p);
        return;
    }

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
 * 
 * The algorithm divides the original set of points into four sets:
 *      - taking 4 points (highest, lowest, rightmos, leftmost) we divide the sets
 *        tracing lines from leftmost to highest, from highest to righmost (we take points above these lines);
 *        from rightmost to lowest, from lowest to leftmost (we take points below these lines)
 *      - each hull is computed for every set
 *      - these 4 hulls are than concatenated into one final hull
 */
void convex_hull(const points_t *pset, points_t *hull)
{
    const int n = pset->n;
    const point_t *p = pset->p;
    int i, j;
    
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
    points_t partial_hulls[4];
    hull->n = 0;
    hull->p = (point_t*)malloc(n * sizeof(point_t)); assert(hull->p);

    /* Main loop that computes hull for each slice of the original set. */
    for (j = 0; j < 4; j++) {
        points_t partial_set;

        /* Divide the original set into partial splitted
           accordingly to the line of two cardinal points. */
        divide_set(pset, cardinal[j], cardinal[(j+1) % 4], &partial_set);

        /* Check partial set capacity. */
        if (partial_set.n > 1) {
            /* Compute the convex hull of the partial set. */
            partial_convex_hull(&partial_set, &partial_hulls[j], 0, partial_set.n - 1);
        } else {
            partial_hulls[j].n = 0;
        }

        /* Free partial set of points. */
        free_pointset(&partial_set);
    }

    /* Add the obtained hull to the final one. */
    for (j = 0; j < 4; j++) {
        for (i = 0; i < partial_hulls[j].n; i++) {
            hull->p[hull->n++] = partial_hulls[j].p[i];
        }

        /* Check partial hull capacity. */
        if (partial_hulls[j].n > 0) {
            /* Free partial hull. */
            free_pointset(&partial_hulls[j]);
        }
    }

    /* Resize the hull with its actual size. */
    //hull->p = (point_t*)realloc(hull->p, hull->n * sizeof(point_t)); assert(hull->p);
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
