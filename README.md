### Parallel_Programming (Open MPI in C)
#### Conjugate Gradient method with stencil-based matrix-vector multiplication

When it comes to modeling of any continuous phenomenon around us, it can be simplified to a linear form as Ax=b.  
There are several ways to solve this matrix equation, but they are all categorized into groups of deterministic or stochastic algorithms.  
On the other hand, many physical problems can be solved by using Stencil computations. It is a great advantage because computers do not need to make and store or read matrix A, so all matrix vector multiplication step would be on the fly. Conjugate gradient as an iterative algorithm can be considered a deterministic method for large positive definite matrices. This robust algorithm combined with stencil benefits and ease of implementation is a good choice compare to Gaussian elimination algorithms.


##### To run the program:  
mpirun -np ‘x’ P_CG_stencil ‘y’  
When x is the number of processes and y is the number of intervals for vector b.
