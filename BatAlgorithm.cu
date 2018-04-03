/////////////////////////////////////////////////
//  nvcc BatAlgorithm.cu
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_profiler_api.h>
#include "cuda.h"
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#define PI 3.14159265

__device__ void Ackley(int D, int i, double* Sol, double* Fitness){
  double a = 20;
  double b = 0.2;
  double c = 2*PI;
  double val_1 = 0.0;
  double val_2 = 0.0;
  for (int j = 0; j < D; j++) {
    val_1 = val_1 + (Sol[(i*D) + j]*Sol[(i*D) + j])/D;
    val_2 = val_2 + cos(c*Sol[(i*D) + j])/D;
  }
  val_1 = sqrt(val_1);
  Fitness[i] = a + exp(1.0) - (a*exp(-b*val_1)) - exp(val_2);
}

__device__ void Schwefel(int D, int i, double* Sol, double* Fitness){
  double val = 0.0;
  for (int j = 0; j < D; j++) {
    val = val + Sol[(i*D) + j]*sin(sqrt(abs(Sol[(i*D) + j])));
  }
  Fitness[i] = (418.9829*D) - val;
}

__device__ void Fun3(int D, int i, double* Sol, double* Fitness){
  double val = 0.0;
  for (int j = 0; j < D; j++) {
    val = val + (Sol[(i*D) + j]*Sol[(i*D) + j]);
  }
  double r = sin(sqrt(val));
  r *= r;
  r = r - 0.5;
  double b = 1.0 + 0.001*val;
  b *= b;
  Fitness[i] = (0.5 - (r/b))*(-1);
}

__device__ void funFitness (int D, int i, double* Sol, double* Fitness, int fun){
  switch (fun) {
    case 1:{
      Ackley(D, i, Sol, Fitness);
      break;}
    case 2:{
      Schwefel(D, i, Sol, Fitness);
      break;}
    case 3:{
      Fun3(D, i, Sol, Fitness);
      break;}
  }
}

__device__ double simplebounds(double val, double lower, double upper){
  if (val < lower) val = lower;
  if (val > upper) val = upper;
  return val;
}

__global__ void best_bat(int D, double* Fitness, double* F, int NP, double* best, double* Sol, int* J,int b){
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  int NumHilos = blockDim.x * gridDim.x;
  F[i] = Fitness[i];
  J[i] = i;
  int ii = i + NumHilos;
  while (ii < (NP - b)){
    if (F[i] > Fitness[ii]){
      F[i] = Fitness[ii];
      J[i] = ii;
    }
    ii += NumHilos;
  }
  __syncthreads();
  if (blockIdx.x == 0) {
    i = threadIdx.x;
    ii = i + blockDim.x ;
    while (ii < NumHilos){
      if (F[i] > F[ii]){
        F[i] = F[ii];
        J[i] = J[ii];
      }
      ii +=  blockDim.x;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      i = 0;
      ii = i + 1 ;
      while (ii < blockDim.x){
        if (F[i] > F[ii]){
          F[i] = F[ii];
          J[i] = J[ii];
        }
        ii ++;
      }
      double td = Fitness[J[i]];
      Fitness[J[i]] = Fitness[NP - b - 1];
      Fitness[NP - b - 1] = td;
      //Fitness[J[i]] = 100000;
      for (size_t j = 0; j < D; j++) {
        best[j + (D*b)] = Sol[(J[i]*D) + j];
        Sol[(J[i]*D) + j] = Sol[((NP - b - 1)*D) + j];
        Sol[((NP - b - 1)*D) + j] = best[j + (D*b)];
      }

    }
  }
}

__global__ void init_bat(int D, double* Lb, double* Ub, double *v, double * Sol, double* Fitness, double* Q, int function){
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState_t state;
  Q[i] = 0;
  curand_init((unsigned long long)clock(), i, 0, &state);
  double rnd;
  for (int j = 0; j < D; j++) {
    rnd = curand_uniform_double(&state);
    v[(i*D) + j] = 0.0;

    Sol[(i*D) + j] = Lb[j] + (Ub[j] - Lb[j])*rnd;
    Sol[(i*D) + j] = simplebounds(Sol[(i*D) + j], Lb[j], Ub[j]);
  }
  funFitness(D, i, Sol, Fitness, function);

  __syncthreads();
}

__global__ void move_bat(int D, double* Lb, double* Ub, double *v, double * Sol,
                    double* Fitness, double* Q, double Qmin, double Qmax, double A,
                    double* best, double* S, double r, double* Fnew, int function){
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState_t state;
  //a[i] = i;
  curand_init((unsigned long long)clock(), i, 0, &state);
  double rnd;
  int k = curand(&state) % 20;
  rnd = curand_uniform_double(&state);
  Q[i] = Qmin + (Qmin - Qmax)*rnd;
  for (int j = 0; j < D; j++) {
    v[(i*D) + j] = v[(i*D) + j] + ((Sol[(i*D) + j] - best[j + (k*D)])*Q[i]);
    S[(i*D) + j] = Sol[(i*D) + j] + v[(i*D) + j];
    S[(i*D) + j] = simplebounds(S[(i*D) + j], Lb[j], Ub[j]);
  }
  rnd = curand_uniform_double(&state);
  if (rnd > r) {

    for (int j = 0; j < D; j++) {
      rnd = curand_uniform_double(&state);
      //rnd = curand_normal_double(&state);
      S[(i*D) + j] = best[j + (k*D)] + (((Ub[j] - Lb[j])/20)* A * ((rnd*2)-1));
      S[(i*D) + j] = simplebounds(S[(i*D) + j], Lb[j], Ub[j]);
    }
  }
  funFitness(D, i, S, Fnew, function);
  rnd = curand_uniform_double(&state);
  if (Fnew[i] <= Fitness[i] && rnd < A) {
    for (int j = 0; j < D; j++) {
      Sol[(i*D) + j] = S[(i*D) + j];
    }
    Fitness[i] = Fnew[i];
  }
  __syncthreads();
}

void run_bat (int D, int NP,int N_Gen, double A, double r, double Qmin, double Qmax, double Lower, double Upper, int function){

  unsigned long long int D_size = D*sizeof(double);
  unsigned long long int NP_size = NP*sizeof(double);
  unsigned long long int DxNP_size = D*NP*sizeof(double);

  double *Lb, *Ub, *best; //size D
  double *Q, *Fnew, *Fitness, *F;      // size NP
  double  *v, *Sol, *S;//size D*NP
  int *J;

  double *_Lb = (double*)malloc(D_size);
  double *_Ub = (double*)malloc(D_size);

  double f, fnew, ff;
  for (int i = 0; i < D; i++) {
    _Lb[i] = Lower;
    _Ub[i] = Upper;
  }
  cudaMallocManaged(&Lb, D_size);
  cudaMallocManaged(&Ub, D_size);
  cudaMallocManaged(&best, D_size*20);

  cudaMallocManaged(&Q, NP_size);
  cudaMallocManaged(&Fnew, NP_size);
  cudaMallocManaged(&Fitness, NP_size);
  cudaMallocManaged(&F, NP_size);

  cudaMallocManaged(&v, DxNP_size);
  cudaMallocManaged(&Sol, DxNP_size);
  cudaMallocManaged(&S, DxNP_size);

  cudaMallocManaged(&J, NP*sizeof(int));

  cudaMemcpy(Lb, _Lb, D_size, cudaMemcpyHostToDevice );
  cudaMemcpy(Ub, _Ub, D_size, cudaMemcpyHostToDevice );
  init_bat<<< NP/100, 100>>>(D, Lb, Ub, v, Sol, Fitness, Q, function);
  for (size_t i = 0; i < 20; i++) {
    best_bat<<< 10, 100>>>(D, Fitness, F, NP, best, Sol, J, i);
    cudaMemcpy(&f, F, sizeof(double), cudaMemcpyDeviceToHost);
    if(i == 0) ff = f;
  }
  //printf("-------------\n");
  //printf("%5.10f\n", ff);

  for (size_t i = 0; i < N_Gen; i++) {
    move_bat<<< NP/100, 100>>>(D, Lb, Ub, v, Sol, Fitness, Q, Qmin, Qmax, A, best, S, r, Fnew, function);
    for (size_t i = 0; i < 20; i++) {
      best_bat<<< 10, 100>>>(D, Fitness, F, NP, best, Sol, J, i);
      cudaMemcpy(&f, F, sizeof(double), cudaMemcpyDeviceToHost);
      if(i == 0) ff = f;
    }
    //printf("-------------\n");
    //printf("%5.10f\n", ff);
    if(fnew < f){
      A *= 0.8;
      r *= (1 - exp(-1.0));
      f = fnew;
    }
  }
  printf("-------------\n");
  printf("%5.10f\n", ff);
  cudaFree(Lb); cudaFree(Ub); cudaFree(best);
  cudaFree(Q); cudaFree(Fnew); cudaFree(Fitness);
  cudaFree(F); cudaFree(v); cudaFree(Sol);
  cudaFree(S); cudaFree(J);
}

int main() {
  for (size_t i = 0; i < 100; i++) {
    //run_bat(2, 20000, 50, 0.5, 0.5, 0.0, 2.0, -32.7680, 32.7680, 1);
    //run_bat(2, 20000, 50, 0.5, 0.5, 0.0, 2.0, -500.0, 500.0, 2);
    run_bat(8, 20000, 50, 0.5, 0.5, 0.0, 2.0, -100.0, 100.0, 3);
  }
  return 0;
}
