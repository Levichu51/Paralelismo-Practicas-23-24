//
// Created by levixhu on 5/04/24.
//

#include <stdio.h>
#include <math.h>
#include <mpi.h>


//Para compilar -> mpicc -o p1 p1.c
//Para ejecutar -> mpirun --oversubscribe -np (nº procs) ./p1


int main(int argc, char *argv[]) {

    int i, done = 0, n;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x, partialSolution;
    int rank, numprocs;

    MPI_Init(&argc, &argv); //Inicializa MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Id de los procs
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs); //Num procs

    while (!done) {
        if (rank == 0) {
            printf("Enter the number of intervals: (0 quits)\n");
            scanf("%d", &n);

            for (int dest = 1; dest < numprocs; dest++) {
                MPI_Send(&n, 1, MPI_INT, dest, 0, MPI_COMM_WORLD); //Envia el intervalo
            }
        } 
        else {
            MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); //Reciben el intervalo del maestro
        }
        
        if (n == 0) {
            done = 1;
        } 
        else {
            h = 1.0 / (double)n;
            sum = 0.0;
            
            for (i = rank + 1; i <= n; i += numprocs) {
                x = h * ((double)i - 0.5);
                sum += 4.0 / (1.0 + x * x);
            }
            
            pi = h * sum;
            
            // Enviar el valor de pi al proceso 0 por parte los procs secundarios al master
            if (rank != 0) {
                MPI_Send(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            } 
            else {
                double recv_pi;

                // Recibir el valor de pi de los otros procesos y sumarlo
                for (int src = 1; src < numprocs; src++) {
                    MPI_Recv(&recv_pi, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    pi += recv_pi;
                }
                // Calcular el error y mostrar el resultado
                partialSolution = pi;

                if(rank == 0){
                    printf("pi is approximately %.16f, Error is %.16f\n", partialSolution, fabs(partialSolution - PI25DT));
                }
            }
        }
    }

    MPI_Finalize();


    return 0;
}

