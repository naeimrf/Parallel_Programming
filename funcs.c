/* ************************************
 * PDP - Individual Project           *
 * Functions for CG with Stencil      *
 * Naeim Rashidfarokhi - June 2020    *
 ************************************ */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* -------------------------------------------------- */

double *load_vector(int n) {
    double x, y;
    int i, j;
    const double h = 1.0 / n;

    double *lv = (double *)calloc((n + 1) * (n + 1), sizeof(double));
    const int half = (n % 2 != 0) ? n / 2 + 1 : n / 2;

    for (i = 1; i < half; i++) {
        for (j = 1; j < n; j++) {
            x = i * h;
            y = j * h;
            lv[i * (n + 1) + j] = lv[(n - i) * (n + 1) + j] =
                2 * h * h * ((x * (1 - x)) + y * (1 - y));
        }
    }
    if (n % 2 == 0) {
        i = half;
        for (j = 1; j < n; j++) {
            x = i * h;
            y = j * h;
            lv[i * (n + 1) + j] = 2 * h * h * ((x * (1 - x)) + y * (1 - y));
        }
    }
    return lv;
}

void print_mesh(double *arr, const int n) {

    for (int j = n + 1; j > 0; j--) {
        for (int i = 0; i < n + 1; i++) {
            printf("%0.4f ", arr[(j - 1) + (n + 1) * i]);
        }
        printf("\n");
    }
}

void print_arr(double *arr, const int n) {

    /*     printf("> Total vector elements:\n");
        for (int i = 0; i < n + 1; i++)
            for (int j = 0; j < n + 1; j++)
                printf("b%d%d: %0.4f\n", i, j, arr[i * (n + 1) + j]); */

    printf("> On hard disk:\n");
    for (int i = 0; i < n + 1; i++) {
        for (int j = 0; j < n + 1; j++)
            printf("%0.2f ", arr[i * (n + 1) + j]);
    }
    printf("\n");
}

/* -------------------------------------------------- */

double *b_inner_points(int n) {
    double x, y;
    int i, j;
    const double h = 1.0 / n;
    double *bip = (double *)malloc((n - 1) * (n - 1) * sizeof(double));
    const int half = ((n - 1) % 2 == 0) ? (n / 2) + 1 : n / 2;

    for (i = 1; i < half; i++) {
        for (j = 1; j < n; j++) {
            x = i * h;
            y = j * h;
            bip[(i - 1) * (n - 1) + (j - 1)] =
                bip[(n - i - 1) * (n - 1) + (j - 1)] =
                    2 * h * h * ((x * (1 - x)) + y * (1 - y));
        }
    }
    if ((n - 1) % 2 != 0) {
        i = half;
        for (j = 1; j < n; j++) {
            x = i * h;
            y = j * h;
            bip[(i - 1) * (n - 1) + (j - 1)] =
                2 * h * h * ((x * (1 - x)) + y * (1 - y));
        }
    }
    return bip;
}

void print_inner_mesh(double *arr, const int n) {

    printf("Inner point corresponds to vector b:\n");
    for (int j = n - 1; j > 0; j--) {
        for (int i = 1; i < n; i++) {
            printf("%0.4f ", arr[(j - 1) + (n - 1) * (i - 1)]);
        }
        printf("\n");
    }
}

void print_arr_inner(double *arr, const int n) {

    /*     printf("> From array:\n");
        for (int i = 1; i < n; i++)
            for (int j = 1; j < n; j++)
                printf("b%d%d: %0.4f\n", i, j, arr[(i - 1) * (n - 1) + (j -
       1)]); */

    // printf("> On hard disk:\n");
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++)
            printf("%0.4f ", arr[(i - 1) * (n - 1) + (j - 1)]);
    }
    printf("\n");
}

/* -------------------------------------------------- */

double *minus_array(double *source, int size) {
    double *arr = (double *)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        arr[i] = -source[i];
    }
    return arr;
}

double transpose_product(double *source, int size) {
    double result = 0;

    for (int i = 0; i < size; i++)
        result += source[i] * source[i];
    return result;
}

double measure_time() {
    struct timespec t0;
    clock_gettime(CLOCK_REALTIME, &t0);
    double time_sec = t0.tv_sec + t0.tv_nsec / 1e9;
    return time_sec;
}

int check_mesh_vs_process(int n, int p) {
    int mesh = (n + 1) * (n + 1);

    while (mesh != 1) {
        /* checks whether a number is divisible by 2 */
        if (mesh % 2 != 0)
            return -1;
        mesh /= 2;
    }

    if (mesh % p == 0) {
        printf("Share of each process:%d", mesh / p);
        return 0;
    } else {
        printf("Imbalanced workload: Share of each process:%0.2f\n",
               (double)mesh / p);
        return -1;
    }
}

void free_resources(double *c, double *d, double *e, double *f, double *g,
                    double *h, double *i, double *j) {

    free(c);
    free(d);
    free(e);
    free(f);
    free(g);

    if (h != NULL)
        free(h);
    if (i != NULL)
        free(i);
    if (j != NULL)
        free(j);
}

void information(int n, int size, int *chunk_sizes) {
    const unsigned int whole_b = (n + 1) * (n + 1);
    printf("Number of internvals(n) in mesh for b:%d\n", n);
    printf("Total number of elements for vector b:%d\n", whole_b);
    printf("Number of inner points: %dx%d=%d\n", (n - 1), (n - 1),
           (n - 1) * (n - 1));

        printf("Number of columns in each process: ");
    for (int i = 0; i < size; i++)
        printf("P%d:%d ", i, chunk_sizes[i]);
    printf("when each column has:%d elements!\n", n + 1);

    if (n < 256)
        printf("The program works best with n: 255, 511, 1023,...!\n");
    const double h = 1.0 / n;
    printf("Intervals(n):%d, h=1/%d:%lf\n* * *\n", n, n, h);
}

/* -------------------------------------------------- */
double *part_of_b(int nbOf_columns, int nbr_elements, int start, int end, int n,
                  int rank, int size) {
    double x, y;
    int i, j, k;
    const double h = 1.0 / n;
    const int column = n + 1;

    if (rank == 0)
        start = k = 1;
    else
        k = 0;
    if (rank == size - 1)
        end = end - 1;

    // printf("Inside part_of_b; start:%d, end:%d, nbr_col:%d\n", start, end,
    // nbOf_columns);
    double *lv = (double *)calloc(nbr_elements, sizeof(double));
    for (i = start; i <= end && k < nbOf_columns; i++, k++) {
        for (j = 1; j < n; j++) {
            x = i * h;
            y = j * h;
            lv[k * column + j] = 2 * h * h * ((x * (1 - x)) + y * (1 - y));
            // printf("%0.4lf ", lv[i * column + j]);
        }
    }
    // printf("\n");
    return lv;
}

void print_portion_mesh(double *arr, int start, int end, int nbOf_columns,
                        const int n, int rank) {

    printf("Portion of arr:? in Rank:%d, start:%d, end:%d, nbr_columns:%d\n",
           rank, start, end, nbOf_columns);
    const int column = n + 1;
    /* j iterate among elements of a column */
    for (int j = column; j > 0; j--) {
        /* i controls column number in each process */
        for (int i = 0; i < nbOf_columns; i++) {
            printf("%0.4f ", arr[column * i + (j - 1)]);
        }
        printf("\n");
    }
}

void print_portion(double *arr, const int n, int nbOf_columns, int rank) {

    const int column = n + 1;
    printf("> Vector elements in rank:%d\n", rank);
    for (int i = 0; i < nbOf_columns + 1; i++)
        for (int j = 0; j < n + 1; j++)
            printf("b%d%d: %0.4f\n", i, j, arr[i * column + j]);

    printf("> Rank:%d elements on hard disk:\n", rank);
    for (int i = 0; i < nbOf_columns + 1; i++) {
        for (int j = 0; j < n + 1; j++)
            printf("%0.4f ", arr[i * column + j]);
    }
    printf("\n");
}

void alone_1PE(int n, int iteration, double *u_portion, double *q_portion,
               double *g_portion, double *d_portion, double *temp, double *norm,
               double *q1, double *q0, double *tau, double *beta,
               double time1) {

    if (n < 4) {
        printf("The program works with n > 3. Please try again!\n");
        exit(-1);
    }

    double time2 = MPI_Wtime();

    const unsigned int whole_b = (n + 1) * (n + 1);
    const unsigned int columns = (n + 1);
    /* Step3: CG */
    int cursor;
    const int half_col = columns / 2;

    for (int it = 0; it < iteration; it++) {
        /* Step4: stencil instead of matrix_vector multiplication */
        for (int i = 1; i < half_col; i++) {
            cursor = i * columns;
            for (int j = 1; j < n; j++) {
                q_portion[cursor + j] = 4 * d_portion[cursor + j] -
                                        d_portion[cursor + (j - 1)] -
                                        d_portion[cursor + (j + 1)] -
                                        d_portion[cursor - columns + j] -
                                        d_portion[cursor + columns + j];
            }
            /* memcpy instead of q[(n * columns) - cursor + j] in j loop
            to copy an entire column to the right based on symmetry */
            memcpy(q_portion + (n * columns) - cursor + 1,
                   q_portion + cursor + 1, (n - 1) * sizeof(double));
        }
        if (columns % 2 != 0) {
            cursor = half_col * columns;
            for (int j = 1; j < n; j++) {
                q_portion[cursor + j] = 4 * d_portion[cursor + j] -
                                        d_portion[cursor + (j - 1)] -
                                        d_portion[cursor + (j + 1)] -
                                        d_portion[cursor - columns + j] -
                                        d_portion[cursor + columns + j];
            }
        }

        /* Step5: first scalar in iterations as tau! */
        *tau = 0;
        for (int j = 0; j < whole_b; j++)
            (*tau) += d_portion[j] * q_portion[j];
        *tau = (*q0) / (*tau);

        /* Step6 & 7: to update vector u & g with loop fusion */
        for (int k = 0; k < whole_b; k++) {
            u_portion[k] = u_portion[k] + (*tau) * d_portion[k];
            g_portion[k] = g_portion[k] + (*tau) * q_portion[k];
        }

        /* Step8: second scalar in iterations as q1! */
        *q1 = 0;
        for (int l = 0; l < whole_b; l++) {
            *temp = g_portion[l];
            *q1 += (*temp) * (*temp);
        }

        /* Step9: another scalar beta1! */
        *beta = (*q1) / (*q0);

        /* Step10: to update vector d! */
        for (int x = 0; x < whole_b; x++)
            d_portion[x] = -1 * g_portion[x] + (*beta) * d_portion[x];

        /* Step11: replacing q1 with q0 for next iteration */
        *q0 = *q1;
    }

    for (int i = 0; i < whole_b; i++) {
        *temp = g_portion[i];
        *norm += (*temp) * (*temp);
    }

    double time3 = MPI_Wtime();

    *norm = sqrt(*norm);
    printf("-----\nNorm of vector g:%f\n-----\n", *norm);

    printf("Time to initialize data: %fs\n", time2 - time1);
    printf("Time for CG algorithm: %fs with %d iterations.\n", time3 - time2,
           iteration);
}