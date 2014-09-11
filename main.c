#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

//#define DEBUG
#define PARALLEL
#define BENCH_EXEC_TIMES 1


// ----------------------------------------------------------
// Type definitions
// ----------------------------------------------------------
typedef struct bench_res {
    int chunk_size;
    double static_sorting_time;
    double static_input_sorting_time;
    double dynamic_sorting_time;
    double dynamic_input_sorting_time;
} bench_res;

/**
 * Represents the action:
 *      0: process a specific file.
 *      1: generates a random file.
 *      2: generates benchmark.
 */
typedef enum action {
    ERROR = -1,
    PROCESS = 0,
    GENERATE = 1,
    BENCHMARK = 2
} action;


// ----------------------------------------------------------
// Type signatures
// ----------------------------------------------------------
int process(float **vec, int *vec_size, double *time_sorting_only, double
        *time_with_input, const char *filename, omp_sched_t kind, int
        chunk_size);
void count_sort(float[], int, omp_sched_t, int);

void generate_floats(FILE* out, int count);
int read_input(FILE* fp, float** vector, int* size);
float* parse_floats(char* values_str, int count);

int bench(const char* filename);
double calculate_mean(const double *vec, int size);
void write_csv(bench_res *res, int size, FILE *fp);

void usage();
action parse_params(int argc, char *argv[], char **filename, int *num_items,
        omp_sched_t *kind, int *chunk_size);


// ----------------------------------------------------------
// Functions
// ----------------------------------------------------------
void count_sort(float a[], int n, omp_sched_t kind, int chunk_size) {
    // According to the documentation, the variables declared before the
    // parallel pragma are *shared*.
    float* temp = malloc(n * sizeof(float));

#ifdef DEBUG
    // This is useful to count the number of iterations each thread calculated,
    // particularlly when using a *dynamic* schedule.
    int *thread_iteration_count, thread_iteration_count_size;
#endif

#ifdef PARALLEL
#pragma omp parallel
    {
        int count; // *Private* variable.

#ifdef DEBUG
#pragma omp single // Serializes allocation of shared array (executed by just one thread).
        {
            thread_iteration_count = (int *) calloc(omp_get_num_threads(), sizeof(int));
            thread_iteration_count_size = omp_get_num_threads();

            // By now, threads should've been created.
            printf("[DEBUG] Current number of threads: %d.\n", omp_get_num_threads());
        }
#endif
        omp_set_schedule(kind, chunk_size);
#pragma omp for schedule(runtime)
#endif
        for (int i = 0; i < n; i++) {
            count = 0;
#ifdef DEBUG
            printf("\t[DEBUG] I'm thread: %d calculating iteration %d...\n", omp_get_thread_num(), i);
            thread_iteration_count[omp_get_thread_num()]++;
#endif
            for (int j = 0; j < n; j++)
                if (a[j] < a[i])
                    count++;
                else if (a[j] == a[i] && j < i)
                    count++;
            // I suppose `temp` is a shared variable, because it was declared before
            // omp's parallel pragma. Considering that `count` will return a
            // different value (indexing a different position of memory) for every
            // thread, I believe it doesn't need to be marked as *critical*.
            temp[count] = a[i];
        }
    }

#ifdef DEBUG
    // "Join" should have occured at this point, so omp_get_num_threads() is expected to return just 1.
    printf("[DEBUG] Current number of threads: %d.\n", omp_get_num_threads()); // assert(1);
    // Prints how many iterations each thread calculated.
    for (int i = 0; i < thread_iteration_count_size; i++)
        printf("\t[DEBUG] Thread: %d calculated %d iterations...\n", i, thread_iteration_count[i]);
    free(thread_iteration_count);
#endif

    memcpy(a, temp, n * sizeof(float));
    free(temp);
} /* count_sort */


// ----------------------------------------------------------
// Benchmark
// ----------------------------------------------------------
double calculate_mean(const double *vec, int size) {
    double res = 0.0;
    for (int i = 0; i < size; i++) {
        res += vec[i];
    }
    return res / size;
}


void write_csv(bench_res *res, int size, FILE *fp) {
    // Writes header... (using ';' for better compatibility)
    fputs("Chunk Size, Static, Static w/ input, Dynamic, Dynamic w/ input\n", fp);

    // Writes body...
    for (int i = 0; i < size; i++)
        fprintf(fp, "%d, %f, %f, %f, %f\n", res[i].chunk_size,
                res[i].static_sorting_time, res[i].static_input_sorting_time,
                res[i].dynamic_sorting_time, res[i].dynamic_input_sorting_time);
}


int bench(const char* filename) {
    FILE *fp;
    double static_sorting_times[BENCH_EXEC_TIMES];
    double static_input_sorting_times[BENCH_EXEC_TIMES];
    double dynamic_sorting_times[BENCH_EXEC_TIMES];
    double dynamic_input_sorting_times[BENCH_EXEC_TIMES];
    bench_res res[6];

    /* static/dynamic, varying chunk size and considering/not considering input
     * time. */
    fp = fopen("bench.csv", "w");

    printf("Looping...\n");
    for (int i = 1, k = 0; i <= 100000; i *= 10, k++) {
        printf("Performing iteration %d, chunk size is %d.\n", k, i);
        for (int j = 0; j < BENCH_EXEC_TIMES; j++) {
            double time_sorting_only, time_with_input;

            // Process using static scheduling.
            if (process(NULL, NULL, &time_sorting_only, &time_with_input, filename, omp_sched_static, i) != 0)
                return -1;
            static_sorting_times[j] = time_sorting_only; 
            static_input_sorting_times[j] = time_with_input;

            // Process using dynamic scheduling.
            if (process(NULL, NULL, &time_sorting_only, &time_with_input, filename, omp_sched_dynamic, i) != 0)
                return -1;
            dynamic_sorting_times[j] = time_sorting_only;
            dynamic_input_sorting_times[j] = time_with_input;
        }
        res[k].chunk_size = i;
        res[k].static_sorting_time = calculate_mean(static_sorting_times, BENCH_EXEC_TIMES);
        res[k].static_input_sorting_time = calculate_mean(static_input_sorting_times, BENCH_EXEC_TIMES);
        res[k].dynamic_sorting_time = calculate_mean(dynamic_sorting_times, BENCH_EXEC_TIMES);
        res[k].dynamic_input_sorting_time = calculate_mean(dynamic_input_sorting_times, BENCH_EXEC_TIMES);
    }

    printf("Generating output CSV file...\n");
    write_csv(res, 6, fp);
    printf("File 'bench.csv' successfully generated! :-)\n");

    fclose(fp);

    return 0;
} /* bench */


// ----------------------------------------------------------
// Util
// ----------------------------------------------------------
float* parse_floats(char* values_str, int count) {
    if (count <= 0) return NULL;

    float *vector = (float *) calloc(count, sizeof(float));

    for (int i = 0; i < count; i++) {
        vector[i] = strtof(values_str, &values_str);
    }

    return vector;
} /* parse_floats */


int read_input(FILE* fp, float** vector, int* size) {
    // Reads vector's length.
    fscanf(fp, "%d", size);
    getc(fp); // Reads '\n'. If it comes from a file, it must be in UNIX format.
    if (size <= 0) return -1;

    char values_str[8000];
    if (fgets(values_str, 8000, fp) == NULL) return -1;

    // Reads vector's contents.
    *vector = parse_floats(values_str, *size);

    return 0;
} /* read_input */


/**
 * Utilitary function to generate a file containing random data in the
 * appropriate format to be processed by the program.
 */
void generate_floats(FILE* out, int count) {
    srand(time(NULL));
    fprintf(out, "%d\n", count);
    for (int i = 0; i < count; i++) {
        float r = (float) rand() / (float) (rand() + 1);
        fprintf(out, "%.2f ", r);
    }
    putc('\n', out);
} /* generate_floats */


void usage() {
    printf("Usage:\n");
    printf("./a.out <action> <params>\n");
    printf("\n");
    printf("Process single file:\n");
    printf("./a.out filename [-k] [-c]\n");
    printf("\n");
    printf("Generate file with random float numbers:\n");
    printf("./a.out -gen filename 10\n");
    printf("\n");
    printf("Benchmark:\n");
    printf("./a.out -bench filename\n");
    printf("\n");
    printf("Options' meanings:\n");
    printf("-k = schedule: {static|dynamic}.\n");
    printf("-c = chunk size: number\n");
}


action parse_params(int argc, char *argv[], char **filename, int *num_items, omp_sched_t *kind, int *chunk_size) {
    if (argc == 2) {
        *filename = argv[1];
        return PROCESS;
    } else if (argc >= 3) {
        if (strcmp(argv[1], "-bench") == 0) {
            *filename = argv[2];
            //if (strcmp(argv[3], "-s") == 0) {
            //   char *schedule = argv[4];
            //   if (strcmp(schedule, "static") == 0) {
            //       *kind = omp_sched_static;
            //   } else if (strcmp(schedule, "dynamic") == 0) {
            //       *kind = omp_sched_dynamic;
            //   }
            //}
            return BENCHMARK;
        } else if (strcmp(argv[1], "-gen") == 0) {
            *filename = argv[2];
            *num_items = atoi(argv[3]);
            return GENERATE;
        } else if (argc == 8) {
            *filename = argv[1];
            if (strcmp(argv[3], "dynamic") == 0)
                *kind = omp_sched_dynamic;
            else
                *kind = omp_sched_static;
            *chunk_size = atoi(argv[5]);

            return PROCESS;
        }
    }
    return ERROR;
}


// ----------------------------------------------------------
// Main
// ----------------------------------------------------------
int process(float **vec, int *vec_size, double *time_sorting_only, double *time_with_input, const char *filename, omp_sched_t kind, int chunk_size) {
     // `chunk_size` 0 or negative to use whatever default is. As I set
     // `omp_sched_auto` before, this value will be ignored anyway (when passed
     // to `omp_set_schedule()`).
    float *vector;
    int vector_size;

    // Measures data reading.
    double t1 = omp_get_wtime();
    FILE* handle = fopen(filename, "r");
    if (handle == NULL) {
        printf("Could not open file: %s.\n", filename);
        return -1;
    }

    if (read_input(handle, &vector, &vector_size) == -1) {
        printf("Invalid input data.\n");
        return -1;
    }

    // Measures sorting only.
    double t2 = omp_get_wtime();
    count_sort(vector, vector_size, kind, chunk_size);
    double t3 = omp_get_wtime();

    // Returns the sorted vector and its size.
    if (vec_size != NULL && vec != NULL) {
        *vec = vector;
        *vec_size = vector_size;
    } else {
        free(vector);
    }

    // Necessary because we make multiple calls to `main` in function `bench()`.
    fclose(handle);

    // Returns elapsed times.
    if (time_sorting_only != NULL)
        *time_sorting_only =  t3 - t2;
    if (time_with_input != NULL)
        *time_with_input = t3 - t1;

    return 0;
}


int main(int argc, char *argv[]) {
    char *filename;
    float *vector;
    FILE *handle;
    int chunk_size = 0, num_items, vector_size;
    omp_sched_t kind = omp_sched_auto;

    // Gets start time.
    double exec_time = omp_get_wtime();

    action act = parse_params(argc, argv, &filename, &num_items, &kind, &chunk_size);

    switch (act) {
        case ERROR:
            usage();
            return -1;
        case GENERATE:
            handle = fopen(filename, "w");
            if (handle == NULL) {
                printf("Could not open file: %s.\n", filename);
                return -1;
            }
            generate_floats(handle, num_items);
            fclose(handle);
            return 0;
        case BENCHMARK:
            if (bench(filename) != 0) {
                printf("It shouldn't happen! :-(\n");
                return -1;
            }
        default:
            if (process(&vector, &vector_size, NULL, NULL, filename, kind, chunk_size) != 0) {
                printf("It shouldn't happen! :-(\n");
                return -1;
            }
    }

    // Gets elapsed time.
    exec_time = omp_get_wtime() - exec_time;

    if (act != BENCHMARK) {
        for (int i = 0; i < vector_size; i++) {
            printf("%.2f ", *(vector+i));
        }
        printf("\n");
    }
    printf ("Execution time: %f seconds.\n", exec_time);

    return 0;
}

