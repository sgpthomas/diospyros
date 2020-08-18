#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xtensa/sim.h>
#include <xtensa/tie/xt_pdxn.h>
#include <xtensa/tie/xt_timer.h>
#include <xtensa/xt_profiling.h>

#include "../../../diospyros-private/src/utils.h"

float a[N * N] __attribute__((section(".dram0.data")));
float q[N * N] __attribute__((section(".dram0.data")));
float r[N * N] __attribute__((section(".dram0.data")));
float q_spec[N * N] __attribute__((section(".dram0.data")));
float r_spec[N * N] __attribute__((section(".dram0.data")));

// For Nature.
float scratch[N] __attribute__((section(".dram0.data")));
float nat_v[(2*N-N+1)*(N/2+N)] __attribute__((section(".dram0.data")));
float nat_d[N] __attribute__((section(".dram0.data")));
float nat_b[N*N] __attribute__((section(".dram0.data")));

// Diospyros kernel
void kernel(float * A, float * Q, float * R);

// Nature functions.
extern "C" {
  void matinvqrf(void *pScr,
                 float32_t* A,float32_t* V,float32_t* D, int M, int N_);
  size_t matinvqrf_getScratchSize(int M, int N_);
  void matinvqrrotf(void *pScr,
                    float32_t* B, const float32_t* V,
                    int M, int N_, int P);
  size_t matinvqrrotf_getScratchSize(int M, int N_, int P);
  void transpmf(const float32_t * x, int M, int N_, float32_t * z);
}

/**
 * QR decomposition using Nature kernels.
 */
int nature_qr(const float *A, float *Q, float *R) {
  // Check that our scratch array is big enough for both steps.
  size_t scratchSize = matinvqrf_getScratchSize(N, N);
  if (sizeof(scratch) < scratchSize) {
    printf("scratch is too small for matinvqrf!\n");
    return 1;
  }
  scratchSize = matinvqrrotf_getScratchSize(N, N, N);
  if (sizeof(scratch) < scratchSize) {
    printf("scratch is too small for matinvqrrotf!\n");
    return 1;
  }

  // Copy A into R (because it's overwritten in the first step).
  for (int i = 0; i < N * N; ++i) {
    R[i] = A[i];
  }

  // Start with main QR decomposition call.
  // The `R` used here is an in/out parameter: A in input, R on output.
  matinvqrf(scratch, R, nat_v, nat_d, N, N);

  // We need an extra identity matrix for the second step.
  zero_matrix(nat_b, N, N);
  for (int i = 0; i < N; ++i) {
    nat_b[i*N + i] = 1;
  }

  // Apply Householder rotations to obtain Q'.
  matinvqrrotf(scratch, nat_b, nat_v, N, N, N);

  // Transpose that to get Q.
  transpmf(nat_b, N, N, Q);
  return 0;
}

float sgn(float v);


// Naive implementation
void naive_transpose(float *a, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = i+1; j < n; j++) {
      float tmp = a[i*n + j];
      a[i*n + j] =  a[j*n + i];
      a[j*n + i] = tmp;
    }
  }
}

float naive_norm(float *x, int m) {
  float sum = 0;
  for (int i = 0; i < m; i++) {
    sum += pow(x[i], 2);
  }
  return sqrt(sum);
}

void naive_matrix_multiply(float *a, float *b, float *c, int row1, int col1, int col2) {
 for (int y = 0; y < row1; y++) {
   for (int x = 0; x < col2; x++) {
     c[col2 * y + x] = 0;
     for (int k = 0; k < col1; k++) {
       c[col2 * y + x] += a[col1 * y + k] * b[col2 * k + x];
     }
   }
 }
}

void naive_qr_decomp(float *A, float *Q, float *R, int n) {
  memcpy(R, A, sizeof(float) * n * n);

  // Build identity matrix of size n * n
  float *I = (float *)calloc(sizeof(float), n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      I[i*n + j] = (i == j);
    }
  }

  // Householder
  for (int k = 0; k < n - 1; k++) {
    int m = n - k;

    float *x = (float *)calloc(sizeof(float), m);
    float *e = (float *)calloc(sizeof(float), m);
    for (int i = 0; i < m; i++) {
      int row = k + i;
      x[i] = R[row*n + k];
      e[i] = I[row*n + k];
    }

    float alpha = -sgn(x[0]) * naive_norm(x, m);

    float *u = (float *)calloc(sizeof(float), m);
    float *v = (float *)calloc(sizeof(float), m);
    for (int i = 0; i < m; i++) {
      u[i] = x[i] + alpha * e[i];
    }
    float norm_u = naive_norm(u, m);
    for (int i = 0; i < m; i++) {
      v[i] = u[i]/norm_u;
    }

    float *q_min = (float *)calloc(sizeof(float), m * m);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float q_min_i = ((i == j) ? 1.0 : 0.0) - 2 * v[i] * v[j];
        q_min[i*m + j] = q_min_i;
      }
    }

    print_matrix(q_min, m, m);
    float *q_t = (float *)calloc(sizeof(float), n * n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        float q_t_i;
        if ((i < k) || (j < k)) {
          q_t_i = (i == j) ? 1.0 : 0.0;
        } else {
          q_t_i =  q_min[(i - k)*m + (j - k)];
        }
        q_t[i*n + j] = q_t_i;
      }
    }

    if (k == 0) {
      memcpy(Q, q_t, sizeof(float) * n * n);     // Q = q_t
      naive_matrix_multiply(q_t, A, R, n, n, n); // R = q_t * A
    } else {
      float *res = (float *)calloc(sizeof(float), n * n);
      naive_matrix_multiply(q_t, Q, res, n, n, n); // R = q_t * A
      memcpy(Q, res, sizeof(float) * n * n);
      naive_matrix_multiply(q_t, R, res, n, n, n); // R = q_t * A
      memcpy(R, res, sizeof(float) * n * n);
    }
    free(x);
    free(e);
    free(u);
    free(v);
    free(q_min);
    free(q_t);
  }
  naive_transpose(Q, n);
}

// Naive with fixed size
void naive_fixed_transpose(float *a) {
  for (int i = 0; i < N; i++) {
    for (int j = i+1; j < N; j++) {
      float tmp = a[i*N + j];
      a[i*N + j] =  a[j*N + i];
      a[j*N + i] = tmp;
    }
  }
}

float naive_fixed_norm(float *x, int m) {
  float sum = 0;
  for (int i = 0; i < m; i++) {
    sum += pow(x[i], 2);
  }
  return sqrt(sum);
}

void naive_fixed_matrix_multiply(float *a, float *b, float *c) {
 for (int y = 0; y < N; y++) {
   for (int x = 0; x < N; x++) {
     c[N * y + x] = 0;
     for (int k = 0; k < N; k++) {
       c[N * y + x] += a[N * y + k] * b[N * k + x];
     }
   }
 }
}

void naive_fixed_qr_decomp(float *A, float *Q, float *R) {
  memcpy(R, A, sizeof(float) * N * N);

  // Build identity matrix of size N * N
  float *I = (float *)calloc(sizeof(float), N * N);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      I[i*N + j] = (i == j);
    }
  }

  // Householder
  for (int k = 0; k < N - 1; k++) {
    int m = N - k;

    float *x = (float *)calloc(sizeof(float), m);
    float *e = (float *)calloc(sizeof(float), m);
    for (int i = 0; i < m; i++) {
      int row = k + i;
      x[i] = R[row*N + k];
      e[i] = I[row*N + k];
    }

    float alpha = -sgn(x[0]) * naive_fixed_norm(x, m);

    float *u = (float *)calloc(sizeof(float), m);
    float *v = (float *)calloc(sizeof(float), m);
    for (int i = 0; i < m; i++) {
      u[i] = x[i] + alpha * e[i];
    }
    float norm_u = naive_fixed_norm(u, m);
    for (int i = 0; i < m; i++) {
      v[i] = u[i]/norm_u;
    }

    float *q_min = (float *)calloc(sizeof(float), m * m);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        float q_min_i = ((i == j) ? 1.0 : 0.0) - 2 * v[i] * v[j];
        q_min[i*m + j] = q_min_i;
      }
    }

    print_matrix(q_min, m, m);
    float *q_t = (float *)calloc(sizeof(float), N * N);
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        float q_t_i;
        if ((i < k) || (j < k)) {
          q_t_i = (i == j) ? 1.0 : 0.0;
        } else {
          q_t_i =  q_min[(i - k)*m + (j - k)];
        }
        q_t[i*N + j] = q_t_i;
      }
    }

    if (k == 0) {
      memcpy(Q, q_t, sizeof(float) * N * N);     // Q = q_t
      naive_fixed_matrix_multiply(q_t, A, R); // R = q_t * A
    } else {
      float *res = (float *)calloc(sizeof(float), N * N);
      naive_fixed_matrix_multiply(q_t, Q, res); // R = q_t * A
      memcpy(Q, res, sizeof(float) * N * N);
      naive_fixed_matrix_multiply(q_t, R, res); // R = q_t * A
      memcpy(R, res, sizeof(float) * N * N);
    }
    free(x);
    free(e);
    free(u);
    free(v);
    free(q_min);
    free(q_t);
  }
  naive_fixed_transpose(Q);
}


int main(int argc, char **argv) {

  FILE *file = fopen(OUTFILE, "w");
  if (file == NULL) file = stdout;

  fprintf(file, "kernel,N,cycles\n");

  init_rand(10);

  create_random_mat(a, N, N);
  zero_matrix(q, N, N);
  zero_matrix(r, N, N);
  zero_matrix(q_spec, N, N);
  zero_matrix(r_spec, N, N);

  print_matrix(a, N, N);

  int err;
  int time = 0;

  printf("Starting spec run\n");
  naive_qr_decomp(a, q_spec, r_spec, N);

  // Naive
  start_cycle_timing;
  naive_qr_decomp(a, q, r, N);
  stop_cycle_timing;
  time = get_time();
  print_matrix(q, N, N);
  print_matrix(r, N, N);
  output_check_abs(q, q_spec, N, N);
  output_check_abs(r, r_spec, N, N);
  zero_matrix(q, N, N);
  zero_matrix(r, N, N);
  printf("Naive : %d cycles\n", time);
  fprintf(file, "%s,%d,%d\n","Naive",N,time);

  // Naive Fixed
  start_cycle_timing;
  naive_fixed_qr_decomp(a, q, r);
  stop_cycle_timing;
  time = get_time();
  print_matrix(q, N, N);
  print_matrix(r, N, N);
  output_check_abs(q, q_spec, N, N);
  output_check_abs(r, r_spec, N, N);
  zero_matrix(q, N, N);
  zero_matrix(r, N, N);
  printf("Naive hard size : %d cycles\n", time);
  fprintf(file, "%s,%d,%d\n","Naive hard size",N,time);

  if (N % 4 == 0) {
    // Nature
    start_cycle_timing;
    err = nature_qr(a, q, r);
    stop_cycle_timing;
    if (err) {
      return err;
    }
    time = get_time();
    print_matrix(q, N, N);
    print_matrix(r, N, N);
    zero_matrix(q, N, N);
    zero_matrix(r, N, N);
    printf("Nature : %d cycles\n", time);
    fprintf(file, "%s,%d,%d\n","Nature",N,time);
  }

  // Diospyros
  start_cycle_timing;
  kernel(a, q, r);
  stop_cycle_timing;
  time = get_time();
  print_matrix(q, N, N);
  print_matrix(r, N, N);
  printf("Diospyros : %d cycles\n", time);
  output_check_abs(q, q_spec, N, N);
  output_check_abs(r, r_spec, N, N);
  fprintf(file, "%s,%d,%d\n","Diospyros",N,time);

  return 0;
}
