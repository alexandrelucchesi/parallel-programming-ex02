#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>
#include <limits.h>
#include <float.h>
#include <sys/types.h>
#include <unistd.h>


typedef unsigned long long int ulli;


double rand_between(unsigned *seed, double min, double max) {
    double x = (double) rand_r(seed) / RAND_MAX;
    return min + x * (max - min);
}


typedef struct thread_data {
    int thread_id;
    ulli num_tosses;
} thread_data;


void* run_thread(void *data) {
    ulli num_tosses = ((thread_data *) data)->num_tosses;

    unsigned int seed = time(NULL) + getpid(); // Must have one different seed per thread.
    ulli *num_hits = (ulli *) malloc(sizeof(ulli));
    *num_hits = 0;
    for (ulli toss = 0; toss < num_tosses; toss++) {
        double x = rand_between(&seed, -1, 1);
        double y = rand_between(&seed, -1, 1);

        long double dist_squared = x * x + y * y;
        if (dist_squared <= 1) (*num_hits)++;
    }

    return num_hits;
}


long double monte_carlo_pi(const ulli num_tosses, int num_threads) {
    ulli num_hits = 0;

    if (num_threads <= 1) {
        thread_data data = (thread_data) { .thread_id = -1, .num_tosses = num_tosses };
        ulli *ret;
        ret = run_thread((void *) &data);
        num_hits = *ret;
        free(ret);
    } else {
        pthread_t threads[num_threads];
        thread_data data[num_threads];
        for (int i = 0; i < num_threads; i++) {
            // Similar to OpenMP static scheduling...
            lldiv_t res = lldiv(num_tosses, num_threads);
            data[i] = (thread_data) { .thread_id = i, .num_tosses = res.quot };
            // First thread calculates the remainder...
            if (i == 0) data[i].num_tosses += res.rem;
            if (pthread_create(&threads[i], NULL, run_thread, (void *) &data[i])) {
                fprintf(stderr, "Error creating thread\n");
                return EXIT_FAILURE;
            }
        }
        
        for (int i = 0; i < num_threads; i++) {
            ulli *ret;
            if (pthread_join(threads[i], (void **) &ret)) {
                fprintf(stderr, "Error joining thread\n");
                return EXIT_FAILURE;
            }
            num_hits += *ret;
            free(ret);
        }
    }

    long double pi = ((long double) 4 * num_hits) / ((long double) num_tosses);
    return pi;
}


int main(int argc, char *argv[]) {
    ulli num_tosses;
    int num_threads;

    scanf("%d %llu", &num_threads, &num_tosses);

    struct timeval start, end;
    // Start
    gettimeofday(&start, NULL);

    long double pi = monte_carlo_pi(num_tosses, num_threads);

    // End
    gettimeofday(&end, NULL);
    ulli elapsed =  (ulli) ((end.tv_sec * 1000000L + end.tv_usec) - (start.tv_sec * 1000000L + start.tv_usec));
    
    printf("%Lf\n", pi);
    printf("%llu\n", elapsed);

    return 0;
}

