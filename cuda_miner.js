const ffi = require('ffi-napi');
const ref = require('ref-napi');

const uint8Array = ref.refType(ref.types.uint8);
const uint32Array = ref.refType(ref.types.uint32);
const uint64 = ref.types.uint64;

const libcuda_miner = ffi.Library('./libcuda_miner', {
    'initialize_cuda': ['int', ['size_t', 'int']],
    'deinitialize_cuda': ['int', []],
    'xelis_hash_cuda': ['int', [uint8Array, 'size_t', uint8Array, 'int']],
    'xelis_hash_cuda_nonce': ['int', [uint8Array, 'pointer', 'size_t', uint8Array, 'pointer', uint8Array, 'int', 'int']]
});

module.exports = {
    initializeCuda: (batchSize, numStates) => {
        return libcuda_miner.initialize_cuda(batchSize, numStates);
    },
    deinitializeCuda: () => {
        return libcuda_miner.deinitialize_cuda();
    },
    xelisHashCuda: (input, totalBatchSize, output, state) => {
        const inputBuffer = Buffer.from(input);
        const outputBuffer = Buffer.alloc(output.length);

        const result = libcuda_miner.xelis_hash_cuda(inputBuffer, totalBatchSize, outputBuffer, state);

        outputBuffer.copy(output);
        return result;
    },
    xelisHashCudaNonce: (baseHeader, nonceStart, batchSize, outputHash, outputNonce, difficulty, gpuId, state) => {
        const baseHeaderBuffer = Buffer.from(baseHeader);
        const nonceStartBuffer = ref.alloc('uint64', nonceStart);
        const outputHashBuffer = Buffer.alloc(outputHash.length);
        const outputNonceBuffer = ref.alloc('uint64', outputNonce);
        const difficultyBuffer = Buffer.from(difficulty);

        const result = libcuda_miner.xelis_hash_cuda_nonce(baseHeaderBuffer, nonceStartBuffer, batchSize, outputHashBuffer, outputNonceBuffer, difficultyBuffer, gpuId, state);

        outputHashBuffer.copy(outputHash);
        outputNonce[0] = outputNonceBuffer.deref();
        return result;
    }
};