#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include <math.h>

#define DEBUG 1

#define N 1024

int main(int argc, char *argv[]) {

  int i, j;
  int k, l;
  int rank, size;
  float *matrix = NULL; //root lo maneja
  float *vector = NULL;
  float *result = NULL; //root lo maneja

  int *sendScatter = NULL;
  int *receiveScatter = NULL;
  int *sendGather = NULL;
  int *receiveGather = NULL;
  
  struct timeval  tv1, tv2, tv3, tv4;
  unsigned long commTime = 0; 
  unsigned long computationTime = 0; 
  unsigned long totalCommTime = 0;
  unsigned long totalCompuTime = 0;

  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  //Cálculo de desplazamientos y envío para Scatterv y Gatherv
  //Se disrtibuye de manera equitativa entre procesos, se añade uno más para asignar las filas restantes
  int rowsPerProc = ((N / size) + ((rank < N % size ? 1 : 0)));

  float *subMatrix = malloc(rowsPerProc * N * sizeof(float));
  float *subRes = malloc(rowsPerProc * sizeof(float));
  vector = malloc(N * sizeof(float));

  if(rank == 0){
    matrix = malloc(N * N * sizeof(float));
    result = malloc(N * sizeof(float));
    sendScatter = malloc(size * sizeof(int));
    receiveScatter = malloc(size * sizeof(int));
    sendGather = malloc(size * sizeof(int));
    receiveGather = malloc(size * sizeof(int));

    /* Inicialización de Matrix y Vector */
    for (k = 0; k < N; k++){
      vector[k] = k;
      for (l = 0; l < N; l++){
        *(matrix + (k * N) + l) = k + l;
      }
    }

    //Cálculo del tamaño de los datos a enviar y recibir
    //Se itera por todos los procs para ver cuántos elementos le toca a cada uno
    for (i = 0; i < size; i++){
      sendScatter[i] = ((N / size) + ((i < N % size ? 1 : 0))) * N; //nº de elementos que hay en un cacho de la matriz
      sendGather[i] = ((N / size) + ((i < N % size ? 1 : 0))); //nº de elementos que se enviarán al root desde cada proc
    }

    receiveGather[0] = 0;
    receiveScatter[0] = 0;

    //Cálculo del desplazamiento
    //Indica dónde comenzarán a almacenarse los datos recibidos
    for (i = 1; i < size; i++){
      receiveScatter[i] =  receiveScatter[i - 1] + sendScatter[i - 1];
      receiveGather[i] = receiveGather[i - 1] + sendGather[i - 1];
    }
  }


  //Obtención del tiempo de comunicación para Scatterv
  gettimeofday(&tv1, NULL);

  //Obtiene los datos a ser distribuídos, nº de elementos a distrubuír por proc, desplazamiento, buffer para almacenar los datos de cada proc, tamaño de la porción
  MPI_Scatterv(matrix, sendScatter, receiveScatter, MPI_FLOAT, subMatrix, (rowsPerProc * N), MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

  gettimeofday(&tv2, NULL);

  commTime = (tv2.tv_sec - tv1.tv_sec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);


  gettimeofday(&tv3, NULL);

  /*Calculo de la matriz*/
  for(k = 0; k < rowsPerProc;k++) {
    subRes[k] = 0;

    for(l = 0; l < N; l++) {
      subRes[k] += *(subMatrix + (k * N) + l) * vector[l];
    }
  }

  gettimeofday(&tv4, NULL);
    
  computationTime = (tv4.tv_usec - tv3.tv_usec)+ 1000000 * (tv4.tv_sec - tv3.tv_sec);

  //Obtención del tiempo de comunicación para Gatherv
  gettimeofday(&tv1, NULL);

  //buffer con los res por proc, nº el de envío por proc, dónde se almacenan los datos por el root, nº elementos a enviar de cada proc, desp para saber donde colocar en result
  MPI_Gatherv(subRes, rowsPerProc, MPI_FLOAT, result, sendGather, receiveGather, MPI_FLOAT, 0, MPI_COMM_WORLD);    

  gettimeofday(&tv2, NULL);

  commTime += (tv2.tv_usec - tv1.tv_usec)+ 1000000 * (tv2.tv_sec - tv1.tv_sec);

  //Recolecta de los tiempos
  MPI_Reduce(&commTime, &totalCommTime, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&computationTime, &totalCompuTime, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


  if(rank == 0){
    if(DEBUG){
      printf("Resultado de la multiplicación: \n");

      for (i = 0; i < N; i++){
        printf(" %f \t ",result[i]);
      }
      printf("\n");

      printf("\n\n");

      printf("Tiempo de comunicación total (s): %lf\n", (double) totalCommTime/1E6);
      printf("Tiempo de computación total (s): %lf\n", (double) totalCompuTime/1E6);
      
    }

    free(matrix);
    free(result);
    free(sendScatter);
    free(sendGather);
    free(receiveGather);
    free(receiveScatter);
  }

  free(vector);
  free(subRes);
  free(subMatrix);

  MPI_Finalize();

  return 0;
}
