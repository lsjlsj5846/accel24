#include "namegen.h"
#include "util.h"

#include <cassert>
#include <math.h>
#include <vector>
#include <string.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// You can modify the data structure as you want
struct Tensor {

  Tensor(std::vector<int> shape_) {
    // No initalization for new Tensors
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
    set_zero();
    CHECK_CUDA(cudaMalloc(&buf_gpu, sizeof(float) * n));
    // buf = (float *)malloc(n * sizeof(float));
  }

  /* Alloc memory */
  Tensor(std::vector<int> shape_, bool _init) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    buf = (float *)malloc(n * sizeof(float));
    set_zero();
    CHECK_CUDA(cudaMalloc(&buf_gpu, sizeof(float) * n));
    if (_init == true){
      CHECK_CUDA(cudaMemcpy(buf_gpu, buf, sizeof(float)*n, 
    cudaMemcpyHostToDevice));
    }
    // buf = (float *)malloc(n * sizeof(float));
  }

  /* Alloc memory and copy */
  Tensor(std::vector<int> shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) {
      shape[i] = shape_[i];
    }

    size_t n = num_elem();
    CHECK_CUDA(cudaMalloc(&buf_gpu, sizeof(float) * n));
    CHECK_CUDA(cudaMemcpy(buf_gpu, buf_, sizeof(float)*n, 
    cudaMemcpyHostToDevice));
    // memcpy(buf, buf_, n * sizeof(float));
  }

  ~Tensor() {
    // if (buf_gpu != nullptr)
    //   CHECK_CUDA(cudaFree(buf_gpu));
    if (buf != nullptr)
      free(buf);
  }

  void set_zero() {
    size_t n = num_elem();
    for (size_t i = 0; i < n; i++)
      buf[i] = 0.0;
  }

  size_t num_elem() {
    size_t sz = 1;
    for (size_t i = 0; i < ndim; i++)
      sz *= shape[i];
    return sz;
  }

  // Pointer to data
  float *buf_gpu = nullptr;
  float *buf = nullptr;

  // Shape of tensor, from outermost dimension to innermost dimension.
  // e.g., {{1.0, -0.5, 2.3}, {4.3, 5.6, -7.8}} => shape = {2, 3}
  size_t ndim = 0;
  size_t shape[4];
};

/* Network parameters */
Tensor *character_embedding;
Tensor *W_ir0, *W_iz0, *W_in0, *W_ir1, *W_iz1, *W_in1;
Tensor *W_hr0, *W_hz0, *W_hn0, *W_hr1, *W_hz1, *W_hn1;
Tensor *b_ir0, *b_iz0, *b_in0, *b_ir1, *b_iz1, *b_in1;
Tensor *b_hr0, *b_hz0, *b_hn0, *b_hr1, *b_hz1, *b_hn1;
Tensor *W_fc, *b_fc;
Tensor *rfloats;

/* input, activations, output */
Tensor *input, *emb_out;
Tensor *hidden0, *hidden1;
Tensor *r0, *r1, *z0, *z1, *n0, *n1, *f, *char_prob;
Tensor *rtmp00, *rtmp01, *rtmp02, *rtmp03, *rtmp04;
Tensor *rtmp10, *rtmp11, *rtmp12, *rtmp13, *rtmp14;
Tensor *ztmp00, *ztmp01, *ztmp02, *ztmp03, *ztmp04;
Tensor *ztmp10, *ztmp11, *ztmp12, *ztmp13, *ztmp14;
Tensor *ntmp00, *ntmp01, *ntmp02, *ntmp03, *ntmp04, *ntmp05;
Tensor *ntmp10, *ntmp11, *ntmp12, *ntmp13, *ntmp14, *ntmp15;
Tensor *htmp00, *htmp01, *htmp02;
Tensor *htmp10, *htmp11, *htmp12;
Tensor *ftmp0;

/* Operations */

/*
 * Embedding
 * input: [1] (scalar)
 * weight: [NUM_CHAR x EMBEDDING_DIM]
 * output: [EMBEDDING_DIM]
 */
void embedding(Tensor *input, Tensor *weight, Tensor *output) {
  size_t n = weight->shape[1];
  for (size_t i = 0; i < n; i++) {
    int x = (int)input->buf[0];
    output->buf[i] = weight->buf[x * n + i];
  }
}

/*
 * Elementwise addition
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_add(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] + input2->buf[i];
  }
}

/*
 * Elementwise (1-x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_oneminus(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 - x;
  }
}

/*
 * Elementwise multiplication
 * input1: [*]
 * input2: [*] (same shape as input1)
 * output: [*] (same shape as input1)
 */
void elemwise_mul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t sn = input1->num_elem();
  for (size_t i = 0; i < sn; i++) {
    output->buf[i] = input1->buf[i] * input2->buf[i];
  }
}

/*
 * Elementwise tanh(x)
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_tanh(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = tanhf(x);
  }
}

/*
 * Elementwise Sigmoid 1 / (1 + exp(-x))
 * input: [*]
 * output: [*] (same shape as input)
 */
void elemwise_sigmoid(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = 1.0 / (1.0 + expf(-x));
  }
}

// __device__ float _gpu_sigmoid(float x){
//   return 1.0 / (1.0 + expf(-x));
// }

__device__ float _gpu_tanh(float x){
  return tanhf(x);
}
/*
 * SGEMV
 * input1: [N x K]
 * input2: [K]
 * output: [N]
 */
void matvec(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t N_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  for (size_t i = 0; i < N_; i++) {
    float c = 0.0;
    for (size_t j = 0; j < K_; j++) {
      c += input1->buf[i * K_ + j] * input2->buf[j];
    }
    output->buf[i] = c;
  }
}

/*
 * SGEMM
 * input1: [M x K]
 * input2: [K x N]
 * output: [M x N]
 */
void matmul(Tensor *input1, Tensor *input2, Tensor *output) {
  size_t M_ = input1->shape[0];
  size_t K_ = input1->shape[1];
  size_t N_ = input2->shape[1];
  for (size_t i = 0; i < M_; i++) {
    for (size_t j = 0; j < N_; j++) {
      float c = 0.0;
      for (size_t k = 0; k < K_; k++) {
        c += input1->buf[i * K_ + k] * input2->buf[k * N_ + j];
      }
      output->buf[i * N_ + j] = c;
    }
  }
}

/*
 * Softmax
 * Normalize the input elements according to its exp value.
 * The result can be interpreted as a probability distribution.
 * input: [*]
 * output: [*], (same shape as input)
 */
void softmax(Tensor *input, Tensor *output) {
  size_t n = input->num_elem();
  float sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    sum += expf(x);
  }
  for (size_t i = 0; i < n; i++) {
    float x = input->buf[i];
    output->buf[i] = expf(x) / sum;
  }
}

/*
 * Sample a random index according to the given probability distribution
 * This function is called at most N*MAX_LEN times. Each call uses a
 * random float in [0,1] to sample an index from the given distribution.
 * input: [NUM_CHAR], probability distribution of the characters
 * rng_seq: [N*MAX_LEN],
 */
int random_select(Tensor *input, Tensor *rng_seq, int rng_offset) {
  float r = rng_seq->buf[rng_offset];
  size_t n = input->num_elem();
  float psum = 0.0;
  for (size_t i = 0; i < n; i++) {
    psum += input->buf[i];
    if (psum > r) {
      return i;
    }
  }
  return n - 1;
}

/*
 * Initialize the model.
 * Do input-independent job here.
 */
void namegen_initialize(int N, char *parameter_fname) {

  /* Only the root process reads the parameter */
 
  size_t parameter_binary_size = 0;
  float *parameter =
      (float *)read_binary(parameter_fname, &parameter_binary_size);

  /* Network parameters */
  character_embedding =
      new Tensor({NUM_CHAR, EMBEDDING_DIM}, parameter + OFFSET0);

  W_ir0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET1);
  W_iz0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET2);
  W_in0 = new Tensor({HIDDEN_DIM, EMBEDDING_DIM}, parameter + OFFSET3);
  W_ir1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET4);
  W_iz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET5);
  W_in1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET6);

  W_hr0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET7);
  W_hz0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET8);
  W_hn0 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET9);
  W_hr1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET10);
  W_hz1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET11);
  W_hn1 = new Tensor({HIDDEN_DIM, HIDDEN_DIM}, parameter + OFFSET12);

  b_ir0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET13);
  b_iz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET14);
  b_in0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET15);
  b_ir1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET16);
  b_iz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET17);
  b_in1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET18);

  b_hr0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET19);
  b_hz0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET20);
  b_hn0 = new Tensor({HIDDEN_DIM}, parameter + OFFSET21);
  b_hr1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET22);
  b_hz1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET23);
  b_hn1 = new Tensor({HIDDEN_DIM}, parameter + OFFSET24);

  W_fc = new Tensor({NUM_CHAR, HIDDEN_DIM}, parameter + OFFSET25);
  b_fc = new Tensor({NUM_CHAR}, parameter + OFFSET26);

  /* input, activations, output, etc. */
  input = new Tensor({1, N});
  emb_out = new Tensor({EMBEDDING_DIM, N});

  hidden0 = new Tensor({HIDDEN_DIM, N}, true);
  hidden1 = new Tensor({HIDDEN_DIM, N}, true);

  r0 = new Tensor({HIDDEN_DIM, N});
  z0 = new Tensor({HIDDEN_DIM, N});
  n0 = new Tensor({HIDDEN_DIM, N});

  r1 = new Tensor({HIDDEN_DIM, N});
  z1 = new Tensor({HIDDEN_DIM, N});
  n1 = new Tensor({HIDDEN_DIM, N});

  f = new Tensor({HIDDEN_DIM, N});

  rfloats = new Tensor({N, MAX_LEN});
  // ftmp0 = new Tensor({NUM_CHAR, N});
  char_prob = new Tensor({NUM_CHAR, N});
  cudaDeviceSynchronize();
  printf("************ Initialization Success!!\n");
}

__global__ void fill_gpu_value(const int N, float *buf_gpu, const float _value){
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= N) return;
    buf_gpu[tidx] = _value;
    // printf("************ %f %d \n", buf_gpu[tidx], tidx);
  }

__global__ void gpu_embedding(const int N, const float *i_buf_gpu, 
                              const float *w_buf_gpu, float *o_buf_gpu){
  int _r = blockIdx.y * blockDim.y + threadIdx.y;
  int _c = blockIdx.x * blockDim.x + threadIdx.x;

  // if (r>= EMBEDDING_DIM || c >= N) return;
  // TODO: vernerable due to the typecasting
  // o_buf_gpu[r * N + c] = w_buf_gpu[(int)(i_buf_gpu[r]) * N + c];
  o_buf_gpu[_r*N + _c] = w_buf_gpu[(int)(i_buf_gpu[_c]) * EMBEDDING_DIM + _r];
}

__global__ void gpu_mmbmmbs(const int N, const int K1, const int K2,
                           float* _W1, float* _X1, float* _b1,
                           float* _W2, float* _X2, float* _b2,
                           float* _output
                           ){
int _r = blockIdx.y * blockDim.y + threadIdx.y;
int _c = blockIdx.x * blockDim.x + threadIdx.x;

float _sum = 0.0;
// if () return;

//TODO: Check!! is ++K right?
// // #pragma unroll 128
for (int k1=0; k1<K1; ++k1) 
{_sum += _W1[_r * K1 + k1] * _X1[k1 * N + _c];}

// // #pragma unroll 128
for (int k2=0; k2<K2; ++k2) 
{_sum += _W2[_r * K2 + k2] * _X2[k2 * N + _c];}
_sum += _b1[_r] + _b2[_r];
_output[_r * N + _c] += 1.0 / (1.0 + expf(-_sum));
}

__global__ void gpu_mmbrmmbt(const int N, const int K1, const int K2,
                           float* _W1, float* _X1, float* _b1,
                           float* _W2, float* _X2, float* _b2,
                           float* _r1, float* _output
                           ){
int _r = blockIdx.y * blockDim.y + threadIdx.y;
int _c = blockIdx.x * blockDim.x + threadIdx.x;

float _sum = 0.0;

//TODO: Check!! is ++K right?
// #pragma unroll 128
for (int _k=0; _k<K2; ++_k) 
{_sum += _W2[_r * K2 + _k] * _X2[_k * N + _c];}
_sum *= _r1[_c];
// #pragma unroll 128
for (int _k=0; _k<K1; ++_k) 
{_sum += _W1[_r * K1 + _k] * _X1[_k * N + _c];}

_sum += _b1[_r] + _b2[_r];
_output[_r * N + _c] += _gpu_tanh(_sum);
}

__global__ void gpu_compute_h(const int N, float* zt, float* nt,
                              float* ht
                           ){
int r = blockIdx.y * blockDim.y + threadIdx.y;
int c = blockIdx.x * blockDim.x + threadIdx.x;

// float sum = 0.0;
// TODO: this function is too simple we can reduce the number of threads
float _zt = zt[r*N + c];
ht[r*N + c] = (1-_zt) * nt[r*N + c] + _zt * ht[r*N + c];
}

__global__ void gpu_linear(const int N, const int K,
                           float* _W, float* _X, float* _b,
                           float* _output
                           ){
int _r = blockIdx.y * blockDim.y + threadIdx.y;
int _c = blockIdx.x * blockDim.x + threadIdx.x;

float _sum = 0.0;

//TODO: Check!! is ++K right?
// #pragma unroll 128
for (int _k=0; _k<K; ++_k) 
{_sum += _W[_r * K + _k] * _X[_k * N + _c];}
_sum += _b[_r];
_output[_r * N + _c] += _sum;
}

__global__ void softmax_kernel(float *input, float *output, int N) {
  // Calculate element-wise exponential and store in shared memory
  __shared__ float L[NUM_CHAR];

  unsigned int tid = threadIdx.x; // local
  unsigned int bn = blockIdx.x; // block number
  L[tid] = 0.0;
  L[tid] = expf(input[tid * N + bn]);

  __syncthreads();
  float _sum = 0.0;
  if (tid==0){
    for (int i=0; i<NUM_CHAR; i++){
      _sum += L[i];
    }
   }

  __syncthreads();
  output[tid * N + bn] = input[tid * N + bn] / _sum;  
}

__global__ void gpu_random_select(int N, int ll, float* cp, float* rf, 
char* output, float* input){
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >=N) return;
  float psum = 0.0;
  for (int i=0; i< NUM_CHAR; i++){
    psum += cp[i*N + tid];
    if (psum > rf[tid * MAX_LEN + ll]){
      output[ll * (MAX_LEN+1) + tid] = (char)i;
      input[tid] = (float)i;
      return ;
    }
  }
  output[ll * (MAX_LEN+1) + tid] = (char)(NUM_CHAR-1);
  input[tid] = (float)(NUM_CHAR - 1);
}

float* _mat_to_print;

/*
 * Generate names.
 * Any input-dependent computation/communication must be done here.
 * N: # of names to generate
 * random_floats: N*MAX_LEN sequence of random floats in [0,1].
 * output: 2D-array of size N x (MAX_LEN+1), allocaetd at main.cpp
 */
void namegen(int N, float *random_floats, char *output) {

  // memcpy(rfloats->buf, random_floats, N * MAX_LEN * sizeof(float));
  CHECK_CUDA(cudaMemcpy(rfloats->buf_gpu, random_floats, N * MAX_LEN * sizeof(float),
                        cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  char* g_output;
  memset(output, 0, N * (MAX_LEN + 1) * sizeof(char));
  CHECK_CUDA(cudaMalloc(&g_output, N * (MAX_LEN + 1) * sizeof(char)));
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());

  /* Generate N names */
  /* Initialize input and hidden vector. */
  /* One hidden vector for each GRU layer */
  // input->buf[0] = SOS;
  // hidden0->set_zero();
  // hidden1->set_zero();
  dim3 gridDim_1((N+1023) / 1024);
  dim3 blockDim_1(1024);
  fill_gpu_value<<<gridDim_1, blockDim_1>>>(N, input->buf_gpu, (float)SOS);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());

  for (int l = 0; l < MAX_LEN; l++) {
    /* Embedding */
    // embedding(input, character_embedding, emb_out);
    dim3 blockDim_2(32,32);
    dim3 gridDim_2( (N + 31)/32, (EMBEDDING_DIM+31)/32);
    gpu_embedding<<<gridDim_2, blockDim_2>>>(N, input->buf_gpu, character_embedding->buf_gpu,
                  emb_out->buf_gpu);    
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
    
    // //////////////////////////////////////////////
    // /* Layer 1: input : emb_out & hid: hidden 0*/
    // //////////////////////////////////////////////
    
    dim3 blockDim_3(32,32);
    dim3 gridDim_3( (N + 31)/32, (HIDDEN_DIM+31)/32);
    gpu_mmbmmbs<<<gridDim_3, blockDim_3>>>(N, EMBEDDING_DIM, HIDDEN_DIM,
                W_ir0->buf_gpu, emb_out->buf_gpu, b_ir0->buf_gpu,
                W_hr0->buf_gpu, hidden0->buf_gpu, b_hr0->buf_gpu,
                r0->buf_gpu);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    if (l==0){
      size_t _to_print_size = N * HIDDEN_DIM;
      CHECK_CUDA(cudaMallocHost(&_mat_to_print, sizeof(float) * _to_print_size));
      CHECK_CUDA(cudaMemcpy(_mat_to_print, r0->buf_gpu, 
      sizeof(float) * _to_print_size, cudaMemcpyDeviceToHost));
      printf("\n");
      for (int _j =10; _j<15; _j++){
        for (int _i=10; _i < 15; _i++){
          printf("%f ,", _mat_to_print[_i *N + _j]);
        }
        printf("\n");
      }
    }

    // dim3 blockDim_4(32,32);
    // dim3 gridDim_4( (N + 31)/32, (HIDDEN_DIM+31)/32);
    
    gpu_mmbmmbs<<<gridDim_3, blockDim_3>>>(N, EMBEDDING_DIM, HIDDEN_DIM,
                W_iz0->buf_gpu, emb_out->buf_gpu, b_iz0->buf_gpu,
                W_hz0->buf_gpu, hidden0->buf_gpu, b_hz0->buf_gpu,
                z0->buf_gpu);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    gpu_mmbrmmbt<<<gridDim_3, blockDim_3>>>(N, EMBEDDING_DIM, HIDDEN_DIM,
                W_in0->buf_gpu, emb_out->buf_gpu, b_in0->buf_gpu,
                W_hn0->buf_gpu, hidden0->buf_gpu, b_hn0->buf_gpu,
                r0->buf_gpu, n0->buf_gpu);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

//     //TODO: is it able to overwrite hidden0?
  gpu_compute_h<<<gridDim_3, blockDim_3>>>(N, z0->buf_gpu, n0->buf_gpu,
                                            hidden0->buf_gpu);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
//     //////////////////////////////////////////////
//     /* Layer 2: input : hidden0 & hid: hidden 1*/
//     //////////////////////////////////////////////

    gpu_mmbmmbs<<<gridDim_3, blockDim_3>>>(N, HIDDEN_DIM, HIDDEN_DIM,
                W_ir1->buf_gpu, hidden0->buf_gpu, b_ir1->buf_gpu,
                W_hr1->buf_gpu, hidden1->buf_gpu, b_hr1->buf_gpu,
                r1->buf_gpu);

    gpu_mmbmmbs<<<gridDim_3, blockDim_3>>>(N, HIDDEN_DIM, HIDDEN_DIM,
                W_iz1->buf_gpu, hidden0->buf_gpu, b_iz1->buf_gpu,
                W_hz1->buf_gpu, hidden1->buf_gpu, b_hz1->buf_gpu,
                z1->buf_gpu);

    gpu_mmbrmmbt<<<gridDim_3, blockDim_3>>>(N, HIDDEN_DIM, HIDDEN_DIM,
                W_in1->buf_gpu, hidden0->buf_gpu, b_in1->buf_gpu,
                W_hn1->buf_gpu, hidden1->buf_gpu, b_hn1->buf_gpu,
                r1->buf_gpu, n1->buf_gpu);

    //TODO: is it able to overwrite hidden0?
    gpu_compute_h<<<gridDim_3, blockDim_3>>>(N, z1->buf_gpu, n1->buf_gpu,
                                              hidden1->buf_gpu);
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
    
//     //////////////////////////////////////////////
//     /* Linear: input : hidden1                  */
//     //////////////////////////////////////////////
gpu_linear<<<gridDim_3, blockDim_3>>>(N, HIDDEN_DIM,
W_fc->buf_gpu, hidden1->buf_gpu, b_fc->buf_gpu,f->buf_gpu);
softmax_kernel<<<N, NUM_CHAR>>>(f->buf_gpu, char_prob->buf_gpu, N);

//     //////////////////////////////////////////////
//     /* Move results to CPU and finalize         */
//     //////////////////////////////////////////////

    dim3 blockDim_rand(1024);
    dim3 gridDim_rand((N+1023)/1024);
    gpu_random_select<<<gridDim_rand, blockDim_rand>>>(
      N, l, char_prob->buf_gpu, rfloats->buf_gpu, g_output, input->buf_gpu
    );
  cudaDeviceSynchronize();
  CHECK_CUDA(cudaGetLastError());
  }
cudaDeviceSynchronize();
CHECK_CUDA(cudaGetLastError());
CHECK_CUDA(cudaMemcpy(output, g_output, N * (MAX_LEN+1) * sizeof(char),
                        cudaMemcpyDeviceToHost));
cudaDeviceSynchronize();
CHECK_CUDA(cudaGetLastError());

}

/*
 * Finalize the model.
 * Although it is not neccessary, we recommend to deallocate and destruct
 * everything you made in namegen_initalize() and namegen().
 */
void namegen_finalize() {

  delete character_embedding;
  delete W_ir0;
  delete W_iz0;
  delete W_in0;
  delete W_ir1;
  delete W_iz1;
  delete W_in1;
  delete W_hr0;
  delete W_hz0;
  delete W_hn0;
  delete W_hr1;
  delete W_hz1;
  delete W_hn1;
  delete b_ir0;
  delete b_iz0;
  delete b_in0;
  delete b_ir1;
  delete b_iz1;
  delete b_in1;
  delete b_hr0;
  delete b_hz0;
  delete b_hn0;
  delete b_hr1;
  delete b_hz1;
  delete b_hn1;
  delete W_fc;
  delete b_fc;
  delete rfloats;

  delete input;
  delete emb_out;
  delete hidden0;
  delete hidden1;
  delete r0;
  delete r1;
  delete z0;
  delete z1;
  delete n0;
  delete n1;
  delete f;
  delete char_prob;
  delete rtmp00;
  delete rtmp01;
  delete rtmp02;
  delete rtmp03;
  delete rtmp04;
  delete rtmp10;
  delete rtmp11;
  delete rtmp12;
  delete rtmp13;
  delete rtmp14;
  delete ztmp00;
  delete ztmp01;
  delete ztmp02;
  delete ztmp03;
  delete ztmp04;
  delete ztmp10;
  delete ztmp11;
  delete ztmp12;
  delete ztmp13;
  delete ztmp14;
  delete ntmp00;
  delete ntmp01;
  delete ntmp02;
  delete ntmp03;
  delete ntmp04;
  delete ntmp05;
  delete ntmp10;
  delete ntmp11;
  delete ntmp12;
  delete ntmp13;
  delete ntmp14;
  delete ntmp15;
  delete htmp00;
  delete htmp01;
  delete htmp02;
  delete htmp10;
  delete htmp11;
  delete htmp12;
  delete ftmp0;
}