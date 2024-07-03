#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include "xelis-hash.cuh"

// Example mining kernel that uses the xelis_hash function
__global__ void mining_kernel(uint8_t *input, uint32_t *output, int nonce_start, int num_nonces) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + nonce_start;
    if (idx < nonce_start + num_nonces) {
        uint8_t data[80];
        // Copy input data and set nonce
        memcpy(data, input, 76); // Assuming the nonce is at the end
        data[76] = (idx >> 24) & 0xFF;
        data[77] = (idx >> 16) & 0xFF;
        data[78] = (idx >> 8) & 0xFF;
        data[79] = idx & 0xFF;

        // Compute hash
        xelis_hash_kernel<<<1, 1>>>(data, output + idx * 8);
    }
}

extern "C" {
    void launch_mining_kernel(uint8_t *input, uint32_t *output, int nonce_start, int num_nonces) {
        uint8_t *d_input;
        uint32_t *d_output;
        size_t inputSize = 80 * sizeof(uint8_t);
        size_t outputSize = num_nonces * 8 * sizeof(uint32_t);

        cudaMalloc(&d_input, inputSize);
        cudaMalloc(&d_output, outputSize);

        cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (num_nonces + blockSize - 1) / blockSize;
        mining_kernel<<<numBlocks, blockSize>>>(d_input, d_output, nonce_start, num_nonces);

        cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
    }
}