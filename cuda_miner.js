const ffi = require('ffi-napi');
const ref = require('ref-napi');

const uint8Array = ref.refType(ref.types.uint8);
const uint32Array = ref.refType(ref.types.uint32);

const libcuda_miner = ffi.Library('./libcuda_miner', {
    'launch_mining_kernel': ['void', [uint8Array, uint32Array, 'int', 'int']],
    'xelis_hash_cuda': ['void', [uint8Array, uint8Array]], // Add the function for hash validation
    'compute_difficulty_target': ['uint32', ['string']] // Add the function for computing difficulty target
});

module.exports = {
    runMiningKernel: (input, output, nonceStart, numNonces) => {
        const inputBuffer = Buffer.from(input);
        const outputBuffer = Buffer.alloc(output.length * 4); // 4 bytes per uint32

        libcuda_miner.launch_mining_kernel(inputBuffer, outputBuffer, nonceStart, numNonces);

        for (let i = 0; i < output.length; i++) {
            output[i] = outputBuffer.readUInt32LE(i * 4);
        }
    },
    computeDifficultyTarget: (difficulty) => {
        return libcuda_miner.compute_difficulty_target(difficulty);
    },
    validateHash: (input, output) => {
        const inputBuffer = Buffer.from(input);
        const outputBuffer = Buffer.alloc(output.length);
        libcuda_miner.xelis_hash_cuda(inputBuffer, outputBuffer);
        return outputBuffer;
    }
};