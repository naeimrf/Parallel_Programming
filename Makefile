###############################################################################
# Makefile for Project, Parallel and Distributed Computing 2020.
# Naeim (nara5324@student.uu.se)
###############################################################################

CC = mpicc
CFLAGS = -std=c99 -Wall --std=gnu99
OFLAGS = -Ofast -march=native -funroll-all-loops #-O3
LIBS = -lm
LDFLAGS = -ffunction-sections # to skip the possible impact of any dead code
DEBUGG = #-g
RM = rm -f
SOURCE = P_CG_stencil.o funcs.o
OUT = P_CG_stencil

$(OUT): $(SOURCE)
	$(CC) $(CFLAGS) $(OFLAGS) $(SOURCE) -o $(OUT) $(LIBS) $(DEBUGG)

P_CG_stencil.o: P_CG_stencil.c
	$(CC) $(CFLAGS) $(OFLAGS) $(LIBS) $(DEBUGG) -c P_CG_stencil.c

funcs.o: funcs.h funcs.c
	$(CC) $(CFLAGS) $(OFLAGS) $(LIBS) $(DEBUGG) -c funcs.c

clean:
	$(RM) $(SOURCE) $(OUT)
