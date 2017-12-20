#include "./c_runtime_api.h"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/* TODO: Your code here */
/* all your GPU kernel code, e.g. matrix_softmax_cross_entropy_kernel */

static const size_t TPB = 512;// threads per block

inline size_t get_array_size(DLArrayHandle arr) {
  size_t n = 1;
  for(int i=0;i<arr->ndim;++i){
    n *= arr->shape[i];
  }

  return n;
}

__global__ void array_set_kernel(float* d_arr, float value, const size_t n){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n){
    d_arr[i] = value;
  }
}

__global__ void broad_cast_to_kernel(const float *d_in, float *d_out, 
                                     const size_t n_in, const size_t n_out){
    const int i= blockIdx.x*blockDim.x + threadIdx.x;
  
    if(i<n_out){
      d_out[i] = d_in[i%n_in];
    }
}

__global__ void reduce_sum_kernel(const float *d_in, float *d_out, 
                                     const size_t n_in, const size_t n_out){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n_out){
    d_out[i] = 0.f;

    for(size_t j=i;j<n_in;j+=n_out){
      d_out[i] += d_in[j];
    }
  }
}

__global__ void matrix_add_kernel(const float *d_in_a, const float *d_in_b, 
                                 float *d_out, const size_t n_out){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n_out){
    d_out[i] = d_in_a[i] + d_in_b[i];
  }
}

__global__ void matrix_add_const_kernel(const float *d_in, float *d_out, 
                                        const float val, const size_t n_out){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n_out){
    d_out[i] = d_in[i] + val;
  }
}

__global__ void matrix_multiply_kernel(const float *d_in_a, const float *d_in_b, 
  float *d_out, const size_t n_out){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n_out){
    d_out[i] = d_in_a[i] * d_in_b[i];
  }
}

__global__ void matrix_multiply_const_kernel(const float *d_in, float *d_out, 
                                        const float val, const size_t n_out){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n_out){
    d_out[i] = d_in[i] * val;
  }
}

__global__ void relu_kernel(const float *d_in, float *d_out, const size_t n_out){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n_out){
    d_out[i] = d_in[i]>0.f ? d_in[i] : 0.f;
  }
}

__global__ void relu_grad_kernel(const float *d_in, const float *d_in_grad,
                                 float *d_out, const size_t n_out){
  const int i= blockIdx.x*blockDim.x + threadIdx.x;

  if(i<n_out){
    d_out[i] = d_in[i]>0.f ? d_in_grad[i] : 0.f;
  }
}

__global__ void matrix_softmax_kernel(const float *d_in, float *d_out, 
                                      const size_t n_out, const size_t n_cols){
  const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < n_out) {
    size_t start = (idx / n_cols) * n_cols;
    float max_val = d_in[start];
    for(size_t i = 0; i < n_cols; ++i) {
      max_val = max(max_val, d_in[start + i]);
    }

    float sum = 0.0;
    for(size_t i = 0; i < n_cols; ++i) {
        sum += exp(d_in[start + i] - max_val);
    }
    d_out[idx] = exp(d_in[idx] - max_val) / sum;
  }
}

// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output) {
  // Dynamic shared memory, size provided at kernel launch.
  extern __shared__ float loss_per_row[];
  // Two dimensional thread blocks.
  int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
  threadIdx.x;
  if (y >= nrow) {
    return;
  }
  input_a += y * ncol;
  input_b += y * ncol;
  float maxval = *input_a;
  // Find max for a row.
  for (int x = 1; x < ncol; ++x) {
    maxval = max(maxval, input_a[x]);
  }
  // Deduct by max for a row, and raise to exp.
  float sum = 0;
  for (int x = 0; x < ncol; ++x) {
    sum += exp(input_a[x] - maxval);
  }
  // Compute per-row loss.
  float loss = 0;
  for (int x = 0; x < ncol; ++x) {
    loss -= input_b[x] * log(exp(input_a[x] - maxval) / sum);
  }
  loss_per_row[y] = loss;
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuArraySet(DLArrayHandle arr, float value) { /* TODO: Your code here */
  float *arr_data = (float *)arr->data;
  size_t n = get_array_size(arr);
  size_t n_block = (n + TPB - 1) / TPB;

  array_set_kernel<<<n_block, TPB>>>(arr_data, value, n);
  return 0;
}

int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  // in(4,5) -> out(3,4,5)
  size_t n_in = get_array_size(input);
  size_t n_out = get_array_size(output);
  assert(n_out%n_in == 0);

  const float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;

  size_t n_block = (n_out + TPB - 1) / TPB;
  broad_cast_to_kernel<<<n_block, TPB>>>(in_data, out_data, n_in, n_out);
  return 0;
}

int DLGpuReduceSumAxisZero(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  // in(3,4,5) -> out(4,5)
  size_t n_in = get_array_size(input);
  size_t n_out = get_array_size(output);
  assert(n_in%n_out == 0);

  const float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;

  size_t n_block = (n_out + TPB - 1) / TPB;
  reduce_sum_kernel<<<n_block, TPB>>>(in_data, out_data, n_in, n_out);
  return 0;
}

int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                              const DLArrayHandle matB, DLArrayHandle output) {
  /* TODO: Your code here */
  // assert(matA->ndim == 2 && matB->ndim == 2 && output->ndim == 2);
  // assert(matA->shape[0] == matB->shape[0] == output->shape[0] &&
  //        matA->shape[1] == matB->shape[1] == output->shape[1]);
  
  const float *matA_data = (float *)matA->data;
  const float *matB_data = (float *)matB->data;
  float *out_data = (float *)output->data;

  size_t n_out = get_array_size(output);
  size_t n_block = (n_out + TPB - 1) / TPB;
  matrix_add_kernel<<<n_block, TPB>>>(matA_data, matB_data, out_data, n_out);
  return 0;
}

int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                     DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == 2 && output->ndim == 2);
  assert(input->shape[0] == output->shape[0] &&
         input->shape[1] == output->shape[1]);
  
  const float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;

  size_t n_out = get_array_size(output);
  size_t n_block = (n_out + TPB - 1) / TPB;
  matrix_add_const_kernel<<<n_block, TPB>>>(in_data, out_data, val, n_out);
  return 0;
}

int DLGpuMatrixElementwiseMultiply(const DLArrayHandle matA,
                                   const DLArrayHandle matB,
                                   DLArrayHandle output) {
  /* TODO: Your code here */
  assert(matA->ndim == 2 && matB->ndim == 2 && output->ndim == 2);
  // assert(matA->shape[0] == matB->shape[0] == output->shape[0] &&
  //        matA->shape[1] == matB->shape[1] == output->shape[1]);
  
  const float *matA_data = (float *)matA->data;
  const float *matB_data = (float *)matB->data;
  float *out_data = (float *)output->data;

  size_t n_out = get_array_size(output);
  size_t n_block = (n_out + TPB - 1) / TPB;
  matrix_multiply_kernel<<<n_block, TPB>>>(matA_data, matB_data, out_data, n_out);
  return 0;
}

int DLGpuMatrixMultiplyByConst(const DLArrayHandle input, float val,
                               DLArrayHandle output) {
  /* TODO: Your code here */
  // assert(input->ndim == 2 && output->ndim == 2);
  // assert(input->shape[0] == output->shape[0] &&
  //        input->shape[1] == output->shape[1]);
  
  const float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;

  size_t n_out = get_array_size(output);
  size_t n_block = (n_out + TPB - 1) / TPB;
  matrix_multiply_const_kernel<<<n_block, TPB>>>(in_data, out_data, val, n_out);
  return 0;
}

int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC) {
  /* TODO: Your code here */
  // Hint: use cublas
  // cublas assume matrix is column major
  assert(matA->ndim == 2 && matB->ndim == 2 && matC->ndim == 2);
  // assert(matA->shape[0] == matB->shape[0] == matC->shape[0] &&
  //        matA->shape[1] == matB->shape[1] == matC->shape[1]);

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {return -1;}
  cublasOperation_t transa = transposeA ? CUBLAS_OP_T: CUBLAS_OP_N;
  cublasOperation_t transb = transposeB ? CUBLAS_OP_T: CUBLAS_OP_N;

  // C = alpha*op(A)op(B) + beta*C
  // op(A) m × k , op(B) k × n and C m × n ,
  // op ( A ) = A   if  transa == CUBLAS_OP_N 
  //            A.T if  transa == CUBLAS_OP_T 
  //            A.H if  transa == CUBLAS_OP_C
  int m = transposeB ? matB->shape[0] : matB->shape[1];
  int n = transposeA ? matA->shape[1] : matA->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];

  const float alpha = 1.f;
  const float beta = 0.f;

  const float *matA_data = (float *)matA->data;
  const float *matB_data = (float *)matB->data;
  float *matC_data = (float *)matC->data;
  
  status = cublasSgemm(handle, transb, transa, m, n, k, &alpha,
              matB_data, matB->shape[1], matA_data, matA->shape[1],  
              &beta, matC_data, m);
  if (status != CUBLAS_STATUS_SUCCESS) {return -1;}

  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {return -1;}
  return 0;
}

int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  assert(input->ndim == output->ndim);
  
  const float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;

  size_t n_out = get_array_size(output);
  size_t n_block = (n_out + TPB - 1) / TPB;
  relu_kernel<<<n_block, TPB>>>(in_data, out_data, n_out);
  return 0;
}

int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output) {
  /* TODO: Your code here */
  // assert(input->ndim == in_grad->ndim == output->ndim);
  
  const float *in_data = (float *)input->data;
  const float *in_grad_data = (float *)in_grad->data;
  float *out_data = (float *)output->data;

  size_t n_out = get_array_size(output);
  size_t n_block = (n_out + TPB - 1) / TPB;
  relu_grad_kernel<<<n_block, TPB>>>(in_data, in_grad_data, out_data, n_out);
  return 0;
}

int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output) {
  /* TODO: Your code here */
  // assert(input->ndim == output->ndim == 2);
  
  const float *in_data = (float *)input->data;
  float *out_data = (float *)output->data;

  size_t n_out = get_array_size(output);
  size_t n_block = (n_out + TPB - 1) / TPB;
  size_t n_cols = input->shape[1];
  matrix_softmax_kernel<<<n_block, TPB>>>(in_data, out_data, n_out, n_cols);
  return 0;
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output) {
  assert(input_a->ndim == 2);
  assert(input_b->ndim == 2);
  assert(output->ndim == 1);
  assert(input_a->shape[0] == input_b->shape[0] &&
         input_a->shape[1] == input_b->shape[1]);
  int nrow = input_a->shape[0];
  // Maximum x- or y-dimension of a block = 1024
  // But we need 'nrow' shared memory, and max shared memory is 48KB.
  // Conservatively allow max 16KB shared memory.
  assert(nrow <= 1024 * 4);
  int ncol = input_a->shape[1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  dim3 threads;
  if (nrow <= 1024) {
    threads.x = nrow;
  } else {
    threads.x = 1024;
    threads.y = (nrow + 1023) / 1024;
  }
  // 1 block, each block with 'threads' number of threads with 'nrow' shared
  // memory size
  matrix_softmax_cross_entropy_kernel<<<1, threads, nrow * sizeof(float)>>>(
      nrow, ncol, input_data_a, input_data_b, output_data);
  return 0;
}
