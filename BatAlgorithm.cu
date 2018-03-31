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

//#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); assert(0); } }

__device__ void Ackley(int D, int i, float* Sol, float* Fitness){
  float a = 20;
  float b = 0.2;
  float c = 2*PI;
  float val_1 = 0.0;
  float val_2 = 0.0;
  for (int j = 0; j < D; j++) {
    val_1 = val_1 + (Sol[(i*D) + j]*Sol[(i*D) + j])/D;
    val_2 = val_2 + cos(c*Sol[(i*D) + j])/D;
  }
  val_1 = sqrt(val_1);
  Fitness[i] = a + exp(1.0) - (a*exp(-b*val_1)) - exp(val_2);
}

__device__ void Schwefel(int D, int i, float* Sol, float* Fitness){
  float val = 0.0;
  for (int j = 0; j < D; j++) {
    val = val + Sol[i]*sin(sqrt(abs(Sol[i])));
  }
  Fitness[i] = 418.9829*D - val;
}

__device__ void Fun3(int D, int i, float* Sol, float* Fitness){
  float val = 0.0;
  for (int j = 0; j < D; j++) {
    val = val + Sol[i]*Sol[i];
  }
  float r = sin(sqrt(val));
  r *= r;
  r = r + 0.5;
  float b = 1.0 + 0.001*val;
  b *= b;
  Fitness[i] = 0.5 - (r/b);
}



__device__ float simplebounds(float val, float lower, float upper){
  if (val < lower) val = lower;
  if (val > upper) val = upper;
  return val;
}

__global__ void best_bat(int D, float* Fitness, float* F, int NP, float* best, float* Sol, int* J){
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  int a = 2;
  F[i] = Fitness[i];
  while (i % a == 0 ) {
    __syncthreads();
    if (i + (a/2) < NP) {
      if (F[i] > F[i + (a/2)]) {
        F[i] = F[i + (a/2)];
        J[i] = i + (a/2);
      }
    }else {break;}
    printf("%f - %d - %d\n", F[i], i, i + (a/2));
    a *= 2;
  }
  __syncthreads();
  if (i == 0){
    for (size_t j = 0; j < D; j++) {
      best[j] = Sol[(J[i]*D) + j];
    }
  }
}

__global__ void init_bat(int D, float* Lb, float* Ub, float *v, float * Sol, float* Fitness, float* Q){
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState_t state;
  Q[i] = 0;
  curand_init((unsigned long long)clock(), i, 0, &state);
  float rnd;
  //printf("\n" );
  for (int j = 0; j < D; j++) {
    rnd = curand_uniform(&state);
    Sol[(i*D) + j] = Lb[j] + (Ub[j] - Lb[j])*rnd;
    //printf("%f _ ", Sol[(i*D) + j]);
  }
  Ackley(D, i, Sol, Fitness);
  //printf("%f - %d\n", Fitness[i], i);
  __syncthreads();
}

__global__ void move_bat(int D, float* Lb, float* Ub, float *v, float * Sol,
                    float* Fitness, float* Q, float Qmin, float Qmax, float A,
                    float* best, float* S, float r, float* Fnew){
  int i = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState_t state;
  //a[i] = i;
  curand_init((unsigned long long)clock(), i, 0, &state);
  float rnd;
  rnd = curand_uniform(&state);
  Q[i] = Qmin + (Qmin - Qmax)*rnd;
  for (int j = 0; j < D; j++) {
    v[(i*D) + j] = v[(i*D) + j] + (Sol[(i*D) + j] - best[j])*Q[i];
    S[(i*D) + j] = Sol[(i*D) + j] + v[(i*D) + j];
    S[(i*D) + j] = simplebounds(S[(i*D) + j], Lb[j], Ub[j]);
  }
  rnd = curand_uniform(&state);
  if (rnd > r) {
    for (int j = 0; j < D; j++) {
      S[(i*D) + j] = best[j] + 0.001 * curand_normal(&state);
      S[(i*D) + j] = simplebounds(S[(i*D) + j], Lb[j], Ub[j]);
    }
  }
  Ackley(D, i, S, Fnew); //falta
  rnd = curand_uniform(&state);
  if (Fnew[i] <= Fitness[i] && rnd < A) {
    for (int j = 0; j < D; j++) {
      Sol[(i*D) + j] = S[(i*D) + j];
    }
    Fitness[i] = Fnew[i];
  }
  printf("%f - %d\n", Fitness[i], i);
  __syncthreads();
}

void run_bat (int D, int NP,int N_Gen, float A, float r, float Qmin, float Qmax, float Lower, float Upper){

  unsigned long long int D_size = D*sizeof(float);
  unsigned long long int NP_size = NP*sizeof(float);
  unsigned long long int DxNP_size = D*NP*sizeof(float);

  float *Lb, *Ub, *best; //size D
  float *Q, *Fnew, *Fitness, *F;      // size NP
  float  *v, *Sol, *S;//size D*NP
  int *J;

  float *_Lb = (float*)malloc(D_size);
  float *_Ub = (float*)malloc(D_size);
  for (int i = 0; i < D; i++) {
    _Lb[i] = Lower;
    _Ub[i] = Upper;
  }
  cudaMallocManaged(&Lb, D_size);
  cudaMallocManaged(&Ub, D_size);
  cudaMallocManaged(&best, D_size);

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
  init_bat<<< NP, 1>>>(D, Lb, Ub, v, Sol, Fitness, Q);
  //cudaMemcpy(_Ub, Ub, D_size, cudaMemcpyDeviceToHost);

  best_bat<<< NP, 1>>>(D, Fitness, F, NP, best, Sol, J);
  for (size_t i = 0; i < N_Gen; i++) {
    move_bat<<< NP, 1>>>(D, Lb, Ub, v, Sol, Fitness, Q, Qmin, Qmax, A, best, S, r, Fnew);
    best_bat<<< NP, 1>>>(D, Fitness, F, NP, best, Sol, J);
  }
  cudaMemcpy(_Ub, Ub, D_size, cudaMemcpyDeviceToHost);
}


int main() {
  run_bat(8, 40, 50, 0.5, 0.5, 0.0, 2.0, -32768.0, 32768.0);
  //printf("fada \n");
  return 0;
}
