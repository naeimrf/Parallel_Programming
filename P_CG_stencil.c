/* ************************************
 * PDP - Individual Project           *
 * Parallel code for CG with Stencil  *
 * Naeim Rashidfarokhi - June 2020    *
 ************************************ */
#include "funcs.h"

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("==== Usage: mpirun -np x P_CG_stencil n ====\n-> x: number of "
               "processes\n-> n: number of intervals (mesh points - 1)\n");
        return 1;
    }

    /* Input parameter */
    const unsigned int n = atoi(argv[1]);
    unsigned int iteration = 200;
    if (n < 200)
        iteration = n;

    /* MPI & C variables (scalars in CG) */
    const unsigned int master = 0;
    unsigned int nbOf_columns = 0, remain = 0, start, end, cursor;
    double q0 = 0, q1 = 0, beta = 0, tau = 0, norm = 0, q0_portion = 0;
    double tau_portion = 0, q1_portion = 0, norm_portion = 0, temp = 0;
    const unsigned int one_column = (n + 1);

    int rank, size, namelen, version, subversion;
    double cg_time, cg_max_time, init_time, init_max_time;
    MPI_Comm my_World = MPI_COMM_WORLD; /* Communicator */
    char process_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Win WIN_left, WIN_right;
    MPI_Info info = MPI_INFO_NULL;
    MPI_Request request; /* for MPI_Ibcast at step 2 */
    MPI_Status status;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(my_World, &size); /* Get the number of processors */
    MPI_Comm_rank(my_World, &rank); /* Get my number */

    /* Get the name of each process, inspired by lecture 7 */
    MPI_Get_version(&version, &subversion);
    MPI_Get_processor_name(process_name, &namelen);
    if (rank == master)
        printf("Rank:%d/%d on node:%s with MPI:%d.%d\n", rank, size,
               process_name, version, subversion);

    init_time = MPI_Wtime();

    /* Workload for each process based on number of columns */
    /* remain part is added to be covered by last process */
    nbOf_columns = one_column / size; /* one_column = (n + 1) */
    if ((one_column % size != 0) && (rank == size - 1)) {
        remain = one_column % size;
        nbOf_columns += remain;
    }
    start = rank * (nbOf_columns - remain);
    end = nbOf_columns + start - 1;

    /* total number of elements in each process */
    const unsigned int nbr_els = one_column * nbOf_columns;

    // const unsigned int whole_b = one_column * one_column;
    // printf("RANK:%d, nbr_els:%d, nbOf_columns:%d whole_b:%d\n", rank,
    // nbr_els, nbOf_columns, whole_b);

    /* Info part, Can be commented out! */
    int all_columns[size];
    MPI_Gather(&nbOf_columns, 1, MPI_INT, all_columns, 1, MPI_INT, master,
               my_World);
    if (rank == master)
        information(n, size, all_columns);

    double *b_portion =
        part_of_b(nbOf_columns, nbr_els, start, end, n, rank, size);
    // print_portion_mesh(b_portion, start, end, nbOf_columns, n, rank);

    /* STEP1: Initializing */
    double *u_portion = (double *)calloc(nbr_els, sizeof(double));
    double *q_portion = (double *)calloc(nbr_els, sizeof(double));
    double *g_portion = minus_array(b_portion, nbr_els);
    double *d_portion = (double *)malloc(nbr_els * sizeof(double));
    memcpy(d_portion, b_portion, nbr_els * sizeof(double));
    // double *u = NULL;

    /* STEP2: Multiplication of transpose and a vector as gT & g */
    q0_portion = 0;
    for (int i = 0; i < nbr_els; i++) {
        // temp = g_portion[i];
        // q0_portion += temp * temp;
        q0_portion += g_portion[i] * g_portion[i];
    }

    /* In case of 1 PE */
    if (size == 1) {
        alone_1PE(n, iteration, u_portion, q_portion, g_portion, d_portion,
                  &temp, &norm, &q1, &q0_portion, &tau, &beta, init_time);
        free_resources(b_portion, u_portion, g_portion, d_portion, q_portion,
                       NULL, NULL, NULL);
        MPI_Finalize();
        return 0;
    }

    MPI_Reduce(&q0_portion, &q0, 1, MPI_DOUBLE, MPI_SUM, master, my_World);
    MPI_Ibcast(&q0, 1, MPI_DOUBLE, master, my_World, &request);
    // printf("RANK:%d, q0:%0.6f\n", rank, q0);

    /* one-sided communication */
    double *win_left_start = 0, *win_right_start = 0;
    int window_size_left, window_size_right;
    double *recv_frm_left = NULL, *recv_frm_right = NULL;

    if (rank == master) {
        win_right_start = d_portion + end * one_column;
        window_size_right = one_column * sizeof(double);
        window_size_left = 0;
        recv_frm_right = (double *)malloc(one_column * sizeof(double));
    }

    else if (rank == size - 1) {
        win_left_start = d_portion;
        window_size_right = 0;
        window_size_left = one_column * sizeof(double);
        recv_frm_left = (double *)malloc(one_column * sizeof(double));
    }

    else {
        win_left_start = d_portion;
        win_right_start = d_portion + (nbOf_columns - 1) * one_column;
        window_size_right = one_column * sizeof(double);
        window_size_left = one_column * sizeof(double);
        recv_frm_right = (double *)malloc(one_column * sizeof(double));
        recv_frm_left = (double *)malloc(one_column * sizeof(double));
    }

    MPI_Win_create(win_left_start, window_size_left, sizeof(double), info,
                   my_World, &WIN_left);
    MPI_Win_create(win_right_start, window_size_right, sizeof(double), info,
                   my_World, &WIN_right);

    init_time = MPI_Wtime() - init_time;

    /* Complition of Ibcast for q0 at step 2 */
    MPI_Wait(&request, &status);

    /* STEP3: CG */
    for (int it = 0; it < iteration; it++) {

        MPI_Win_fence((MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE), WIN_left);
        MPI_Win_fence((MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE), WIN_right);

        if (rank == master) { /* saved to ngbr_right! */
            MPI_Get(recv_frm_right, one_column, MPI_DOUBLE, 1, 0, one_column,
                    MPI_DOUBLE, WIN_left);
        }

        else if (rank == size - 1) {
            MPI_Get(recv_frm_left, one_column, MPI_DOUBLE, size - 2, 0,
                    one_column, MPI_DOUBLE, WIN_right);
        }

        else {
            /* to get data from left neighbour */
            MPI_Get(recv_frm_left, one_column, MPI_DOUBLE, rank - 1, 0,
                    one_column, MPI_DOUBLE, WIN_right);

            /* to get data from right neighbour */
            MPI_Get(recv_frm_right, one_column, MPI_DOUBLE, rank + 1, 0,
                    one_column, MPI_DOUBLE, WIN_left);
        }
        MPI_Win_fence(MPI_MODE_NOSUCCEED, WIN_left);
        MPI_Win_fence(MPI_MODE_NOSUCCEED, WIN_right);

        /* STEP4: stencil instead of matrix_vector multiplication */
        const int half_col =
            one_column % 2 == 0 ? (one_column / 2) : (one_column / 2) + 1;
        if (rank == 0) {
            int i, j;
            for (i = 1; i < nbOf_columns - 1; i++) {
                cursor = i * one_column;
                for (j = 1; j < half_col; j++) {
                    q_portion[cursor + j] = q_portion[cursor + (n - j)] =
                        4 * d_portion[cursor + j] -
                        d_portion[cursor + (j - 1)] -
                        d_portion[cursor + (j + 1)] -
                        d_portion[cursor - one_column + j] -
                        d_portion[cursor + one_column + j];
                }
            }
            i = nbOf_columns - 1;
            cursor = i * one_column;
            for (j = 1; j < half_col; j++) {
                q_portion[cursor + j] = q_portion[cursor + (n - j)] =
                    4 * d_portion[cursor + j] - d_portion[cursor + (j - 1)] -
                    d_portion[cursor + (j + 1)] -
                    d_portion[cursor - one_column + j] - recv_frm_right[j];
            }
        }

        else if (rank == size - 1) {
            int j, i = 0;
            cursor = i * one_column;
            for (j = 1; j < half_col; j++) {
                q_portion[cursor + j] = q_portion[cursor + (n - j)] =
                    4 * d_portion[cursor + j] - d_portion[cursor + (j - 1)] -
                    d_portion[cursor + (j + 1)] - recv_frm_left[j] -
                    d_portion[cursor + one_column + j];
            }
            for (i = 1; i < nbOf_columns - 1; i++) {
                cursor = i * one_column;
                for (j = 1; j < half_col; j++) {
                    q_portion[cursor + j] = q_portion[cursor + (n - j)] =
                        4 * d_portion[cursor + j] -
                        d_portion[cursor + (j - 1)] -
                        d_portion[cursor + (j + 1)] -
                        d_portion[cursor - one_column + j] -
                        d_portion[cursor + one_column + j];
                }
            }
        }

        else {
            int j, i = 0;
            cursor = i * one_column;
            for (j = 1; j < half_col; j++) {
                q_portion[cursor + j] = q_portion[cursor + (n - j)] =
                    4 * d_portion[cursor + j] - d_portion[cursor + (j - 1)] -
                    d_portion[cursor + (j + 1)] - recv_frm_left[j] -
                    d_portion[cursor + one_column + j];
            }
            for (i = 1; i < nbOf_columns - 1; i++) {
                cursor = i * one_column;
                for (j = 1; j < half_col; j++) {
                    q_portion[cursor + j] = q_portion[cursor + (n - j)] =
                        4 * d_portion[cursor + j] -
                        d_portion[cursor + (j - 1)] -
                        d_portion[cursor + (j + 1)] -
                        d_portion[cursor - one_column + j] -
                        d_portion[cursor + one_column + j];
                }
            }
            i = nbOf_columns - 1;
            cursor = i * one_column;
            for (j = 1; j < half_col; j++) {
                q_portion[cursor + j] = q_portion[cursor + (n - j)] =
                    4 * d_portion[cursor + j] - d_portion[cursor + (j - 1)] -
                    d_portion[cursor + (j + 1)] -
                    d_portion[cursor - one_column + j] - recv_frm_right[j];
            }
        }

        /* STEP5: first scalar in iterations as tau! */
        tau_portion = 0;
        for (int j = 0; j < nbr_els; j++)
            tau_portion += d_portion[j] * q_portion[j];

        MPI_Reduce(&tau_portion, &tau, 1, MPI_DOUBLE, MPI_SUM, master,
                   my_World);
        MPI_Bcast(&tau, 1, MPI_DOUBLE, master, my_World);
        tau = q0 / tau;

        /* STEP6 & 7: to update vector u & g with loop fusion */
        for (int k = 0; k < nbr_els; k++) {
            u_portion[k] = u_portion[k] + tau * d_portion[k];
            g_portion[k] = g_portion[k] + tau * q_portion[k];
        }

        /* STEP8: second scalar as q1! */
        q1_portion = 0;
        for (int l = 0; l < nbr_els; l++) {
            // temp = g_portion[l];
            // q1_portion += temp * temp;
            q1_portion += g_portion[l] * g_portion[l];
        }

        MPI_Reduce(&q1_portion, &q1, 1, MPI_DOUBLE, MPI_SUM, master, my_World);
        MPI_Bcast(&q1, 1, MPI_DOUBLE, master, my_World);

        /* STEP9: another scalar beta1! */
        beta = q1 / q0;

        /* STEP10: to update vector d in each process! */
        for (int x = 0; x < nbr_els; x++)
            d_portion[x] = -1 * g_portion[x] + beta * d_portion[x];

        /* STEP11: replacing q1 with q0 for next iteration */
        q0 = q1;
    }

    for (int i = 0; i < nbr_els; i++) {
        // temp = g_portion[i];
        // norm_portion += temp * temp;
        norm_portion += g_portion[i] * g_portion[i];
    }

    cg_time = MPI_Wtime() - init_time;

    MPI_Reduce(&norm_portion, &norm, 1, MPI_DOUBLE, MPI_SUM, master, my_World);
    if (rank == master) {
        norm = sqrt(norm);
        printf("-----\nNorm of vector g:%f\n-----\n", norm);
    }

    MPI_Reduce(&init_time, &init_max_time, 1, MPI_DOUBLE, MPI_MAX, master,
               my_World);
    MPI_Reduce(&cg_time, &cg_max_time, 1, MPI_DOUBLE, MPI_MAX, master,
               my_World);

    if (rank == master) {
        printf("Initialization of data: %fs\n", init_max_time);
        printf("CG algorithm: %fs with %d iterations.\n", cg_max_time,
               iteration);
    }

    /* The last NULL pointer reserved for u */
    free_resources(b_portion, u_portion, g_portion, d_portion, q_portion,
                   recv_frm_left, recv_frm_right, NULL);
    MPI_Win_free(&WIN_left);
    MPI_Win_free(&WIN_right);
    MPI_Finalize();
    return 0;
}
