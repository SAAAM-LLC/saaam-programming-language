/*
SAAAM Language - CUDA Kernel Library
No external deps. Optional acceleration layer for SAAAM runtime + compiler backends.
*/

#include "saaam_native_runtime.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static inline bool saaam_cuda_ok(cudaError_t err, const char* call, const char* file, int line) {
    if (err == cudaSuccess) return true;
    fprintf(stderr, "CUDA failure: %s at %s:%d: %s\n", call, file, line, cudaGetErrorString(err));
    return false;
}

#define CUDA_OK(call) saaam_cuda_ok((call), #call, __FILE__, __LINE__)

// ---- Ternary Neural Kernels ----

__global__ void saaam_ternary_matmul_kernel(
    const int8_t* __restrict__ weights, // M*K (ternary)
    const float* __restrict__ input,    // K*N
    float* __restrict__ output,         // M*N
    int M, int N, int K
) {
    int row = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    const int w_row = row * K;
    for (int k = 0; k < K; k++) {
        int8_t w = weights[w_row + k];
        float x = input[k * N + col];
        sum += (w == 1) ? x : (w == -1) ? -x : 0.0f;
    }
    output[row * N + col] = sum;
}

// ---- Attention Kernel (online softmax, no score buffer) ----

__global__ void saaam_concept_attention_kernel(
    const float* __restrict__ queries,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    float* __restrict__ output,
    int num_concepts,
    int embedding_dim
) {
    int q = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (q >= num_concepts || embedding_dim <= 0) return;

    extern __shared__ float sh_partial[]; // size == blockDim.x, reduction buffer

    float qd = (tid < embedding_dim) ? queries[q * embedding_dim + tid] : 0.0f;
    float inv_sqrt_dim = rsqrtf((float)embedding_dim);

    float m = -1.0e30f;
    float s = 0.0f;
    float o = 0.0f;

    for (int k = 0; k < num_concepts; k++) {
        float partial = 0.0f;
        if (tid < embedding_dim) partial = qd * keys[k * embedding_dim + tid];
        sh_partial[tid] = partial;
        __syncthreads();

        for (int offset = (int)blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) sh_partial[tid] += sh_partial[tid + offset];
            __syncthreads();
        }

        float score = sh_partial[0] * inv_sqrt_dim;
        float m_new = fmaxf(m, score);
        float alpha = expf(m - m_new);
        float beta = expf(score - m_new);
        s = s * alpha + beta;
        if (tid < embedding_dim) o = o * alpha + beta * values[k * embedding_dim + tid];
        m = m_new;
        __syncthreads();
    }

    if (tid < embedding_dim) output[q * embedding_dim + tid] = o / s;
}

// ---- Activation / Normalization Kernels ----

__device__ __forceinline__ float saaam_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float saaam_gelu(float x) {
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
        case SAAAM_ACT_RELU: x = fmaxf(0.0f, x); break;
        case SAAAM_ACT_TANH: x = tanhf(x); break;
        case SAAAM_ACT_GELU: x = saaam_gelu(x); break;
        case SAAAM_ACT_SILU: x = saaam_silu(x); break;
        default: break;
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

extern "C" {

static int saaam_next_pow2(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

bool saaam_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

bool saaam_cuda_check_device(void) {
    int device_count = 0;
    if (!CUDA_OK(cudaGetDeviceCount(&device_count))) return false;
    return device_count > 0;
}

bool saaam_cuda_ternary_matmul(const int8_t* weights, const float* input, float* output, int M, int N, int K) {
    if (!weights || !input || !output) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;

    int8_t* d_weights = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;

    bool ok = true;
    ok = ok && CUDA_OK(cudaMalloc(&d_weights, (size_t)M * (size_t)K * sizeof(int8_t)));
    ok = ok && CUDA_OK(cudaMalloc(&d_input, (size_t)K * (size_t)N * sizeof(float)));
    ok = ok && CUDA_OK(cudaMalloc(&d_output, (size_t)M * (size_t)N * sizeof(float)));

    ok = ok && CUDA_OK(cudaMemcpy(d_weights, weights, (size_t)M * (size_t)K * sizeof(int8_t), cudaMemcpyHostToDevice));
    ok = ok && CUDA_OK(cudaMemcpy(d_input, input, (size_t)K * (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    if (ok) {
        dim3 block(16, 16);
        dim3 grid((unsigned)((N + (int)block.x - 1) / (int)block.x),
                  (unsigned)((M + (int)block.y - 1) / (int)block.y));
        saaam_ternary_matmul_kernel<<<grid, block>>>(d_weights, d_input, d_output, M, N, K);
        ok = ok && CUDA_OK(cudaGetLastError());
        ok = ok && CUDA_OK(cudaDeviceSynchronize());
    }

    ok = ok && CUDA_OK(cudaMemcpy(output, d_output, (size_t)M * (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

    if (d_weights) cudaFree(d_weights);
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    return ok;
}

bool saaam_cuda_concept_attention(
    const float* queries,
    const float* keys,
    const float* values,
    float* output,
    int num_concepts,
    int embedding_dim
) {
    if (!queries || !keys || !values || !output) return false;
    if (num_concepts <= 0 || embedding_dim <= 0) return false;

    float *d_queries = nullptr, *d_keys = nullptr, *d_values = nullptr, *d_output = nullptr;
    size_t bytes = (size_t)num_concepts * (size_t)embedding_dim * sizeof(float);

    bool ok = true;
    ok = ok && CUDA_OK(cudaMalloc(&d_queries, bytes));
    ok = ok && CUDA_OK(cudaMalloc(&d_keys, bytes));
    ok = ok && CUDA_OK(cudaMalloc(&d_values, bytes));
    ok = ok && CUDA_OK(cudaMalloc(&d_output, bytes));

    ok = ok && CUDA_OK(cudaMemcpy(d_queries, queries, bytes, cudaMemcpyHostToDevice));
    ok = ok && CUDA_OK(cudaMemcpy(d_keys, keys, bytes, cudaMemcpyHostToDevice));
    ok = ok && CUDA_OK(cudaMemcpy(d_values, values, bytes, cudaMemcpyHostToDevice));

    if (ok) {
        int block = saaam_next_pow2(embedding_dim);
        if (block < 32) block = 32;
        if (block > 256) block = 256;
        dim3 grid((unsigned)num_concepts, 1, 1);
        dim3 threads((unsigned)block, 1, 1);
        size_t shmem = (size_t)block * sizeof(float);

        saaam_concept_attention_kernel<<<grid, threads, shmem>>>(d_queries, d_keys, d_values, d_output, num_concepts, embedding_dim);
        ok = ok && CUDA_OK(cudaGetLastError());
        ok = ok && CUDA_OK(cudaDeviceSynchronize());
    }

    ok = ok && CUDA_OK(cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost));

    if (d_queries) cudaFree(d_queries);
    if (d_keys) cudaFree(d_keys);
    if (d_values) cudaFree(d_values);
    if (d_output) cudaFree(d_output);
    return ok;
}

bool saaam_cuda_activation_inplace(float* data, size_t n, saaam_activation_t act) {
    if (!data || n == 0) return false;

    float* d_data = nullptr;
    bool ok = CUDA_OK(cudaMalloc(&d_data, n * sizeof(float)));
    ok = ok && CUDA_OK(cudaMemcpy(d_data, data, n * sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (int)((n + (size_t)block - 1) / (size_t)block);

    if (ok) {
        if (act != SAAAM_ACT_RMSNORM_SILU) {
            saaam_activation_inplace_kernel<<<grid, block>>>(d_data, n, (int)act);
            ok = ok && CUDA_OK(cudaGetLastError());
            ok = ok && CUDA_OK(cudaDeviceSynchronize());
        } else {
            const float eps = 1.0e-6f;
            float* buf_a = nullptr;
            float* buf_b = nullptr;
            size_t cur_n = n;

            size_t blocks = (cur_n + (size_t)block - 1) / (size_t)block;
            ok = ok && CUDA_OK(cudaMalloc(&buf_a, blocks * sizeof(float)));
            ok = ok && CUDA_OK(cudaMalloc(&buf_b, blocks * sizeof(float)));

            const float* in = d_data;
            float* out = buf_a;
            while (ok) {
                blocks = (cur_n + (size_t)block - 1) / (size_t)block;
                saaam_sumsq_kernel<<<(unsigned)blocks, block>>>(in, out, cur_n);
                ok = ok && CUDA_OK(cudaGetLastError());
                ok = ok && CUDA_OK(cudaDeviceSynchronize());
                if (!ok) break;
                if (blocks == 1) break;
                cur_n = blocks;
                in = out;
                out = (out == buf_a) ? buf_b : buf_a;
            }

            float sumsq = 0.0f;
            ok = ok && CUDA_OK(cudaMemcpy(&sumsq, out, sizeof(float), cudaMemcpyDeviceToHost));
            if (ok) {
                float inv_rms = rsqrtf(sumsq / (float)n + eps);
                saaam_rmsnorm_silu_kernel<<<grid, block>>>(d_data, n, inv_rms);
                ok = ok && CUDA_OK(cudaGetLastError());
                ok = ok && CUDA_OK(cudaDeviceSynchronize());
            }

            if (buf_a) cudaFree(buf_a);
            if (buf_b) cudaFree(buf_b);
        }
    }

    ok = ok && CUDA_OK(cudaMemcpy(data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    if (d_data) cudaFree(d_data);
    return ok;
}

} // extern "C"
