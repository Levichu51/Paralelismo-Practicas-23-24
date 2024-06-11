//
// Created by levixhu on 5/04/24.
//

#include <stdio.h>
#include <math.h>
#include <mpi.h>

//Para compilar -> mpicc -o p2 p2.c -lm
//Para ejecutar -> mpirun --oversubscribe -np (nº procs) ./p2


int MPI_FlattreeColectiva(double *pi, double *partialSolution, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    int rank, numprocs, error;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numprocs); 

    if(partialSolution == NULL){
        return MPI_ERR_BUFFER;
    }
    if(count < 0){
        return MPI_ERR_COUNT;
    }
    if(comm == NULL){
        return MPI_ERR_COMM;
    }
    if(root < 0 || root >= numprocs){
        return MPI_ERR_ROOT;
    }
    if(datatype != MPI_DOUBLE){
        return MPI_ERR_TYPE;
    }

    double recv_pi = *pi;

    //Manejo de errores MPI

    // Se envía el valor de pi desde todos los procesos a 0
    error = MPI_Send(&recv_pi, count, datatype, root, 0, comm);

    if(error != MPI_SUCCESS){
        return error;
    }
    
    
    if (rank == root) {
        *partialSolution = recv_pi;

        // El proceso 0 recibe los valores de pi de todos los demás procesos y los suma
        for (int src = 1; src < numprocs; src++) {
            error = MPI_Recv(&recv_pi, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
            
            if(error != MPI_SUCCESS){
                return error;
            }
            
            *partialSolution += recv_pi;
        }
    }
    
    return error;
}



// Función para la difusión de datos utilizando un árbol binomial
int MPI_BinomialBcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
    int rank, numprocs, error;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &numprocs);

    if(buffer == NULL){
        return MPI_ERR_BUFFER;
    }
    if(count < 0){
        return MPI_ERR_COUNT;
    }
    if(comm == NULL){
        return MPI_ERR_COMM;
    }
    if(root < 0 || root >= numprocs){
        return MPI_ERR_ROOT;
    }
    if(datatype != MPI_INT){
        return MPI_ERR_TYPE;
    }


    for (int i = 1; i <= ceil(log2(numprocs)); i++) {
        
        if (rank < pow(2, i)) { //para participar el rank del proc ha de ser menor que numprocs del nivel
            int paso = pow(2, (i - 1));

            if(rank < paso){  //enviar proceso a un rango superior          
                int dest = rank + paso;
                
                if (dest < numprocs) {
                    error = MPI_Send(buffer, count, datatype, dest, 0, comm);

                    if(error != MPI_SUCCESS){
                        return error;
                    }
                }
            }
            else {
                int src = rank - paso;

                error = MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);

                if(error != MPI_SUCCESS){
                    return error;
                }
            }
        }
    }

    return error;
}



int main(int argc, char *argv[]) {
    int i, done = 0, n;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x, partialSolution;
    int rank, numprocs, error;

    MPI_Init(&argc, &argv); //Inicializa el entorno para la comunicación entre procs
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //Obtener los ids
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs); //Obtener el numero de procs

    while (!done) {

        if (rank == 0) { //Se encaraga el proc principal (el root)
            printf("Enter the number of intervals: (0 quits)\n");
            scanf("%d", &n);
        }
        
        //error = MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD); //Transmite el valor de n desde el proc root a todos los demás
        error = MPI_BinomialBcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if(error != MPI_SUCCESS){
            printf("Bcast ERROR\n");
            MPI_Abort(MPI_COMM_WORLD, error);
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
            
            //error = MPI_Reduce(&pi, &partialSolution, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); //Las soluciones de cada proc se recolectan con esta función para obtener el res final
            error = MPI_FlattreeColectiva(&pi, &partialSolution, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if(error != MPI_SUCCESS){
                printf("Reduce ERROR\n");
                MPI_Abort(MPI_COMM_WORLD, error);
            }

            if (rank == 0) { //El proc root se encarga de esto también
                printf("pi is approximately %.16f, Error is %.16f\n", partialSolution, fabs(partialSolution - PI25DT));
            }
        }
    }

    MPI_Finalize(); //Limpia los recursos usados y cierra la comunicación


    return 0;
}

