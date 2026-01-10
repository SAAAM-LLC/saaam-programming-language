/*
SAAAM Language - CUDA Kernels
GPU-ACCELERATED NEUROPLASTIC PROCESSING

Implements massively parallel:
- Neuroplastic morphing operations
- Ternary neural network operations
- Concept embedding transformations
- Event processing pipelines
*/

#include "saaam_native_runtime.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// === CUDA ERROR CHECKING ===

static inline bool saaam_cuda_ok(cudaError_t err, const char* call, const char* file, int line) {
    if (err == cudaSuccess) return true;
    fprintf(stderr, "CUDA failure: %s at %s:%d: %s\n", call, file, line, cudaGetErrorString(err));
    return false;
}

#define CUDA_OK(call) saaam_cuda_ok((call), #call, __FILE__, __LINE__)

// === TERNARY NEURAL NETWORK KERNELS ===

__device__ int8_t ternary_weight_encode(float weight) {
    // Encode float weight to ternary {-1, 0, 1}
    if (weight > 0.33f) return 1;
    if (weight < -0.33f) return -1;
    return 0;
}

__device__ float ternary_weight_decode(int8_t weight) {
    // Decode ternary weight to float
    return (float)weight;
}

__global__ void saaam_ternary_matmul_kernel(
    const int8_t* __restrict__ weights,    // Ternary weights
    const float* __restrict__ input,       // Input activations
    float* __restrict__ output,            // Output activations
    int M, int N, int K                    // Dimensions: M x K @ K x N
) {
    // Optimized ternary matrix multiplication
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Ternary multiplication is just addition/subtraction!
        for (int k = 0; k < K; k++) {
            int8_t w = weights[row * K + k];
            float inp = input[k * N + col];

            // Ternary multiplication
            if (w == 1) {
                sum += inp;
            } else if (w == -1) {
                sum -= inp;
            }
            // w == 0: skip (no multiplication needed!)
        }

        output[row * N + col] = sum;
    }
}

// === CONCEPT EMBEDDING PROCESSING ===

__global__ void saaam_concept_embedding_transform_kernel(
    const float* __restrict__ concept_embeddings,  // Input embeddings
    const float* __restrict__ transform_matrix,    // Transformation matrix
    float* __restrict__ output_embeddings,         // Output embeddings
    int num_concepts,                              // Number of concepts
    int embedding_dim                              // Embedding dimension
) {
    int concept_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (concept_idx >= num_concepts) return;

    // Transform embedding through neural network layer
    for (int out_dim = 0; out_dim < embedding_dim; out_dim++) {
        float sum = 0.0f;

        for (int in_dim = 0; in_dim < embedding_dim; in_dim++) {
            sum += concept_embeddings[concept_idx * embedding_dim + in_dim] *
                   transform_matrix[in_dim * embedding_dim + out_dim];
        }

        // ReLU activation
        output_embeddings[concept_idx * embedding_dim + out_dim] = fmaxf(0.0f, sum);
    }
}

// === NEURAL ATTENTION MECHANISM ===

__global__ void saaam_concept_attention_kernel(
    const float* __restrict__ queries,      // Query embeddings
    const float* __restrict__ keys,         // Key embeddings
    const float* __restrict__ values,       // Value embeddings
    float* __restrict__ output,             // Output embeddings
    int num_concepts,                       // Number of concepts
    int embedding_dim                       // Embedding dimension
) {
    int q = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    if (q >= num_concepts) return;
    if (embedding_dim <= 0) return;

    extern __shared__ float sh_partial[]; // size == blockDim.x

    float qd = (tid < embedding_dim) ? queries[q * embedding_dim + tid] : 0.0f;
    float inv_sqrt_dim = rsqrtf((float)embedding_dim);

    // Online softmax accumulation per output dimension (one thread per dim).
    float m = -1.0e30f;
    float s = 0.0f;
    float o = 0.0f;

    for (int k = 0; k < num_concepts; k++) {
        float partial = 0.0f;
        if (tid < embedding_dim) {
            partial = qd * keys[k * embedding_dim + tid];
        }

        sh_partial[tid] = partial;
        __syncthreads();

        // Reduce sh_partial[0..blockDim.x) to sh_partial[0]. Assumes power-of-two block size.
        for (int offset = (int)blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                sh_partial[tid] += sh_partial[tid + offset];
            }
            __syncthreads();
        }

        float score = sh_partial[0] * inv_sqrt_dim;
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);

        s = s * alpha + beta;
        if (tid < embedding_dim) {
            float vd = values[k * embedding_dim + tid];
            o = o * alpha + beta * vd;
        }
        m = m_new;
        __syncthreads();
    }

    if (tid < embedding_dim) {
        output[q * embedding_dim + tid] = o / s;
    }
}

// === ACTIVATION / NORMALIZATION KERNELS ===

__device__ __forceinline__ float saaam_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float saaam_gelu(float x) {
    // Hendrycks & Gimpel approximation
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
}

__global__ void saaam_activation_inplace_kernel(float* data, size_t n, int act) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= n) return;

    float x = data[idx];
    switch (act) {
        case SAAAM_ACT_RELU:
            x = fmaxf(0.0f, x);
            break;
        case SAAAM_ACT_TANH:
            x = tanhf(x);
            break;
        case SAAAM_ACT_GELU:
            x = saaam_gelu(x);
            break;
        case SAAAM_ACT_SILU:
            x = saaam_silu(x);
            break;
        default:
            break;
    }
    data[idx] = x;
}

__global__ void saaam_sumsq_kernel(const float* __restrict__ x, float* __restrict__ partial, size_t n) {
    __shared__ float sh[256];
    size_t tid = (size_t)threadIdx.x;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + tid;

    float v = 0.0f;
    if (idx < n) {
        float a = x[idx];
        v = a * a;
    }
    sh[tid] = v;
    __syncthreads();

    for (size_t offset = (size_t)blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) sh[tid] += sh[tid + offset];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sh[0];
}

__global__ void saaam_rmsnorm_silu_kernel(float* x, size_t n, float inv_rms) {
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= n) return;
    float v = x[idx] * inv_rms;
    x[idx] = saaam_silu(v);
}

// === TERNARY LOGIC OPERATIONS (PARALLEL) ===

__global__ void saaam_ternary_and_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int8_t* __restrict__ result,
    size_t count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) return;

    int8_t val_a = a[idx];
    int8_t val_b = b[idx];

    // Ternary AND: {-1, 0, 1} representing {FALSE, UNKNOWN, TRUE}
    if (val_a == -1 || val_b == -1) {
        result[idx] = -1;  // FALSE
    } else if (val_a == 1 && val_b == 1) {
        result[idx] = 1;   // TRUE
    } else {
        result[idx] = 0;   // UNKNOWN
    }
}

__global__ void saaam_ternary_or_kernel(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    int8_t* __restrict__ result,
    size_t count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) return;

    int8_t val_a = a[idx];
    int8_t val_b = b[idx];

    // Ternary OR
    if (val_a == 1 || val_b == 1) {
        result[idx] = 1;   // TRUE
    } else if (val_a == -1 && val_b == -1) {
        result[idx] = -1;  // FALSE
    } else {
        result[idx] = 0;   // UNKNOWN
    }
}

// === HOST-SIDE WRAPPERS ===

extern "C" {

void saaam_cuda_ternary_matmul(
    const int8_t* weights,
    const float* input,
    float* output,
    int M, int N, int K
) {
    // Allocate device memory
    int8_t* d_weights;
    float* d_input;
    float* d_output;

    CUDA_CHECK(cudaMalloc(&d_weights, M * K * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_input, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, M * N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_weights, weights, M * K * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, input, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    saaam_ternary_matmul_kernel<<<gridSize, blockSize>>>(
        d_weights, d_input, d_output, M, N, K
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
}

void saaam_cuda_concept_attention(
    const float* queries,
    const float* keys,
    const float* values,
    float* output,
    int num_concepts,
    int embedding_dim
) {
    // Allocate device memory
    float *d_queries, *d_keys, *d_values, *d_output;
    size_t embedding_size = num_concepts * embedding_dim * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_queries, embedding_size));
    CUDA_CHECK(cudaMalloc(&d_keys, embedding_size));
    CUDA_CHECK(cudaMalloc(&d_values, embedding_size));
    CUDA_CHECK(cudaMalloc(&d_output, embedding_size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_queries, queries, embedding_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_keys, keys, embedding_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, values, embedding_size, cudaMemcpyHostToDevice));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (num_concepts + blockSize - 1) / blockSize;

    saaam_concept_attention_kernel<<<gridSize, blockSize>>>(
        d_queries, d_keys, d_values, d_output,
        num_concepts, embedding_dim
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output, d_output, embedding_size, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_queries);
    cudaFree(d_keys);
    cudaFree(d_values);
    cudaFree(d_output);
}

void saaam_cuda_check_device() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return;
    }

    printf("🚀 Found %d CUDA device(s)\n", device_count);

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        printf("   Device %d: %s\n", i, prop.name);
        printf("     Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("     Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("     Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("     Max threads per block: %d\n", prop.maxThreadsPerBlock);
    }
}

} // extern "C"
