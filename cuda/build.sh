nvcc -Xcompiler -fPIC -shared -o ../libcuda_miner.so miner.cu xelis-hash.cu keccak.cu