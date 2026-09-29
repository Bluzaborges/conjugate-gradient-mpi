#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define IMAX 100
#define ERROR 0.00000001

float **create_matrix_A(int order){
	
	int i;
	float **A = (float**)calloc(order, sizeof(float *));
	for (i = 0; i < order; i++)
		A[i] = (float *)calloc(order, sizeof(float));	
	
	for (i = 0; i < order; i++){
		A[i][i] = 4;
		
		if (i + 1 < order)
			A[i + 1][i] = A[i][i + 1] = 1;
	}
	
	return A;
}

float *create_array_b(int size){

	float *b = (float *)malloc(size * sizeof(float));
	
	for (int i = 0; i < size; i++)
		b[i] = 1;
		
	return b;
}

float *create_array_x(int size){
	return 	(float *)calloc(size, sizeof(float));
}

void matrix_by_array_product_MPI(float **matrix, float *array, float *product, int order, int id, int numberProcesses){

	int startLine = id * order / numberProcesses;
	int finalLine = startLine + order / numberProcesses;

	for (int i = startLine, iLocal = 0; i < finalLine; i++, iLocal++){
		product[iLocal] = 0;
		for (int j = 0; j < order; j++)
			product[iLocal] += array[j] * matrix[i][j];
	}
}

void array_subtraction(float *array1, float *array2, float *difference, int size){
	for (int i = 0; i < size; i++)
		difference[i] = array1[i] - array2[i];
}

void array_sum(float *array1, float *array2, float *sum, int size){
	for (int i = 0; i < size; i++)
		sum[i] = array1[i] + array2[i];
}

float scalar_array_product(float *array1, float *array2, int size){

	float product = 0;
	
	for (int i = 0; i < size; i++)
		product += array1[i] * array2[i];
		
	return product;
}

void scalar_by_array_product(float scalar, float *array, float *product, int size){
	for (int i = 0; i < size; i++)
		product[i] = array[i] * scalar;
}

int conjugate_gradient(float **A, float *b, float *x, int order, int id, int numberProcesses, double *time){
	
	int i = 0;
	float *vaux = (float *)malloc(order * sizeof(float));
	float *q = (float *)malloc(order * sizeof(float));
	float *d = (float *)malloc(order * sizeof(float));
	float *r = (float *)malloc(order * sizeof(float));
	float alpha, newSigma, oldSigma, beta;
	
	float *buffer = (float *)malloc(order / numberProcesses * sizeof(float));
	
	double startTime, finalTime;
	
	startTime = MPI_Wtime();
	
	matrix_by_array_product_MPI(A, x, buffer, order, id, numberProcesses);
	
	MPI_Allgather(buffer, order / numberProcesses, MPI_FLOAT, vaux, order / numberProcesses, MPI_FLOAT, MPI_COMM_WORLD);
	
	array_subtraction(b, vaux, r, order);
	
	memcpy(d, r, order * sizeof(float));
	
	newSigma = scalar_array_product(r, r, order);
	
	while (i < IMAX && newSigma > ERROR){
		
		matrix_by_array_product_MPI(A, d, buffer, order, id, numberProcesses);
		
		MPI_Allgather(buffer, order / numberProcesses, MPI_FLOAT, q, order / numberProcesses, MPI_FLOAT, MPI_COMM_WORLD);
		
		alpha = newSigma / scalar_array_product(d, q, order);
		
		scalar_by_array_product(alpha, d, vaux, order);
		
		array_sum(x, vaux, x, order);
		
		scalar_by_array_product(alpha, q, vaux, order);
		
		array_subtraction(r, vaux, r, order);
		
		oldSigma = newSigma;
		
		newSigma = scalar_array_product(r, r, order);
		
		beta = newSigma / oldSigma;
		
		scalar_by_array_product(beta, d, vaux, order);
		
		array_sum(r, vaux, d, order);
		
		i++;
	}
	
	finalTime = MPI_Wtime();
	
	*time = finalTime - startTime;
	
	free(vaux);
	free(q);
	free(d);
	free(r);
	
	return i;
}

int main(int argc, char **argv){

	MPI_Init(&argc, &argv);

	if (argc != 2){
		printf("%s <matrix order>\n", argv[0]);
		exit(0);
	}
	
	int order = atol(argv[1]), iterations, id, numberProcesses;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &numberProcesses);
	MPI_Status status;
	
	float **A = create_matrix_A(order);
	float *b = create_array_b(order);
	float *x = create_array_x(order);
	
	double time;
	
	iterations = conjugate_gradient(A, b, x, order, id, numberProcesses, &time);
	
	if (id == 0)
		printf("Time: %.4f | Number of iterations: %i\n", time, iterations);
		
	MPI_Finalize();
}
