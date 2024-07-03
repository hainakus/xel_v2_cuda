const StratumClient = require('./stratum');
const ffi = require('ffi-napi');
const ref = require('ref-napi');
const BigNumber = require("bignumber.js");
const uint8Array = ref.refType(ref.types.uint8);
const uint32Array = ref.refType(ref.types.uint32);

const libcuda_miner = ffi.Library('./libcuda_miner', {
    'launch_mining_kernel': ['void', [uint8Array, uint32Array, 'int', 'int']],
});

function computeDifficultyTarget(difficulty) {
    const diff = new BigNumber(difficulty);
    const maxValue = new BigNumber(2).pow(256).minus(1); // 2^256 - 1
    return maxValue.dividedBy(diff);
}
// Convert hex string to byte array
function hexToBytes(hex) {
    const bytes = [];
    for (let i = 0; i < hex.length; i += 2) {
        bytes.push(parseInt(hex.substr(i, 2), 16));
    }
    return bytes;
}
// Convert a hash (array of bytes) to a BigNumber
function fromBigEndian(bytes) {
    let result = new BigNumber(0);
    for (const byte of bytes) {
        result = result.multipliedBy(256).plus(byte);
    }
    return result;
}

// Check if a given hash meets the target difficulty
function checkDifficultyAgainstTarget(hash, target) {
    const hashWork = fromBigEndian(hash);
    console.log(hashWork.toNumber())
    console.log(target.toNumber())
    return hashWork.isLessThanOrEqualTo(target);
}
class Miner {
    constructor(host, port, username, password) {
        this.client = new StratumClient(host, port, username, password);
        this.client.on('job', (params) => this.mine(params));
        this.client.connect();
    }

    async mine(params) {
        const [jobId, prevHash, coinbase1, coinbase2, merkleBranch, version, nBits, nTime, cleanJobs] = params;
        console.log('Received new job:', jobId);

        const nonceStart = 0;
        const numNonces = 1024;
        const input = Buffer.alloc(80); // Adjust size based on actual input data structure
        input.write(prevHash, 0, 'hex');
        input.write(coinbase1, 32, 'hex');
        input.write(coinbase2, 64, 'hex');

        const output = new Array(numNonces * 8).fill(0); // Adjust size according to your hash output
        console.log('Using GPU for mining...');
        libcuda_miner.launch_mining_kernel(input, output, nonceStart, numNonces);

        const difficultyTarget = computeDifficultyTarget(nBits.toString());
        let validNonce = null;
        let validResult = null;

        for (let i = 0; i < numNonces; i++) {
            const hash = output.slice(i * 8, (i + 1) * 8);
            if (checkDifficultyAgainstTarget(hash, difficultyTarget)) {
                validNonce = nonceStart + i;
                validResult = this.getHashResult(hash);
                break;
            }
        }

        if (validNonce !== null) {
            console.log(`Valid nonce found: ${validNonce}, result: ${validResult}`);
            this.client.submit(jobId, validNonce.toString(16), validResult);
        } else {
            console.log('No valid nonce found in this batch');
        }
    }



    getHashResult(hash) {
        return hash.toString('hex'); // Placeholder, replace with actual logic
    }
}

module.exports = Miner;