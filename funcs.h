/* ************************************
 * PDP - Individual Project           *
 * Header file for funcs  CG-Stencil  *
 * Naeim Rashidfarokhi - June 2020    *
 ************************************ */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void print_arr(double *arr, const int n);
void print_arr_inner(double *arr, const int n);
void print_mesh(double *arr, const int n);
void print_inner_mesh(double *arr, const int n);

double *b_inner_points(int n);
double *load_vector(int n);

double *minus_array(double *__restrict source, int size);
double transpose_product(double *source, int size);
int check_mesh_vs_process(int n, int p);
void free_resources(double *c, double *d, double *e, double *f, double *g,
                    double *h, double *i, double *j);
void information(int n, int size, int *chunk_sizes);

double *part_of_b(int nbOf_columns, int nbr_elements, int start, int end, int n,
                  int rank, int size);
void print_portion_mesh(double *arr, int start, int end, int nbOf_columns,
                        const int n, int rank);
void print_portion(double *arr, const int n, int nbOf_columns, int rank);
void alone_1PE(int n, int iteration, double *u_portion, double *q_portion,
               double *g_portion, double *d_portion, double *temp, double *norm,
               double *q1, double *q0, double *tau, double *beta, double time1);