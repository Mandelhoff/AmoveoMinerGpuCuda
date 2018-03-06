#define VERSION_STRING "1.0.0.3"
#define TOOL_NAME "AmoveoMinerGpuCuda"

#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <string>
#include <cassert>

#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <functional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sha256.cuh"

#include <iostream>
#include <chrono>
#include <cmath>
#include <thread>
#include <iomanip>
#include <string>
#include <cassert>

#include <future>
#include <numeric>
#include <chrono>

#include <cpprest/asyncrt_utils.h>

#include "poolApi.h"
#include "base64.h"

using namespace std;
using namespace std::chrono;

using namespace utility;									// Common utilities like string conversions

#define FETCH_WORK_INTERVAL_MS 9000
#define SHOW_INTERVAL_MS 2000

int gElapsedMilliSecMax = FETCH_WORK_INTERVAL_MS;

//#define POOL_URL "http://localhost:32371/work"	// local pool
#define POOL_URL "http://amoveopool.com/work"
#define MINER_ADDRESS "BPA3r0XDT1V8W4sB14YKyuu/PgC6ujjYooVVzq1q1s5b6CAKeu9oLfmxlplcPd+34kfZ1qx+Dwe3EeoPu0SpzcI="
#define DEFAULT_DEVICE_ID 0

string gMinerPublicKeyBase64(MINER_ADDRESS);
string gPoolUrl(POOL_URL);
string_t gPoolUrlW;
int gDevicdeId = DEFAULT_DEVICE_ID;

// Output string by the device read by host
unsigned char *g_out = nullptr;
unsigned char *g_hash_out = nullptr;
int *g_found = nullptr;
unsigned char *g_nonce = nullptr;

static uint64_t totalNonce = 0;
static uint32_t totalSharesFound = 0;

// First timestamp when program starts
static std::chrono::high_resolution_clock::time_point t1;

// Last timestamp we printed debug info
static std::chrono::high_resolution_clock::time_point t_last_updated;
static std::chrono::high_resolution_clock::time_point t_last_work_fetch;



__device__ bool checkResult(unsigned char* h, size_t diff) {
	unsigned int x = 0;
	unsigned int y[2];
	for (int i = 0; i < 31; i++) {
		if (h[i] == 0) {
			x += 8;
			y[1] = h[i + 1];
			continue;
		}
		else if (h[i] < 2) {
			x += 7;
			y[1] = (h[i] * 128) + (h[i + 1] / 2);
		}
		else if (h[i] < 4) {
			x += 6;
			y[1] = (h[i] * 64) + (h[i + 1] / 4);
		}
		else if (h[i] < 8) {
			x += 5;
			y[1] = (h[i] * 32) + (h[i + 1] / 8);
		}
		else if (h[i] < 16) {
			x += 4;
			y[1] = (h[i] * 16) + (h[i + 1] / 16);
		}
		else if (h[i] < 32) {
			x += 3;
			y[1] = (h[i] * 8) + (h[i + 1] / 32);
		}
		else if (h[i] < 64) {
			x += 2;
			y[1] = (h[i] * 4) + (h[i + 1] / 64);
		}
		else if (h[i] < 128) {
			x += 1;
			y[1] = (h[i] * 2) + (h[i + 1] / 128);
		}
		else {
			y[1] = h[i];
		}
		break;
	}
	y[0] = x;
	return(((256 * y[0]) + y[1]) >= diff);
}

__global__ void sha256_kernel(unsigned char* out_input_string_nonce, int *out_found, const unsigned char* in_input_string, size_t in_input_string_size, size_t blockDifficulty, size_t shareDifficulty, const unsigned char * pNonce) {
	__shared__ unsigned char bhash[32];
	__shared__ unsigned char baseNonce[24];
	__shared__ unsigned char diffA;
	__shared__ unsigned char diffB;
	__shared__ SHA256_CTX ctxShared;
	__shared__ size_t blockDiff;
	__shared__ size_t shareDiff;

	// If this is the first thread of the block, init the input string in shared memory
	if (threadIdx.x == 0) {
		memcpy(bhash, in_input_string, in_input_string_size);
		memcpy(baseNonce, pNonce, 24);

		blockDiff = blockDifficulty;
		diffA = blockDiff / 256;
		diffB = blockDiff % 256;
		shareDiff = shareDifficulty;

		sha256_init(&ctxShared);
		sha256_update(&ctxShared, bhash, 32);
		sha256_update(&ctxShared, &diffA, 1);
		sha256_update(&ctxShared, &diffB, 1);
		sha256_update(&ctxShared, baseNonce, 24);
	}

	__syncthreads(); // Ensure the input string has been written in SMEM

	uint64_t currentBlockIdx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned char shaResult[32];
	SHA256_CTX ctx;
	memcpy(&ctx, &ctxShared, 0x70);// sizeof(SHA256_CTX) = 0x70;

	sha256_update(&ctx, (BYTE*)&currentBlockIdx, 8);
	sha256_final(&ctx, shaResult);

	if (checkResult(shaResult, shareDiff) && atomicExch(out_found, 1) == 0) {
		//memcpy(out_found_hash, shaResult, 32);
		memcpy(out_input_string_nonce, baseNonce, 24);
		memcpy(out_input_string_nonce+24, (BYTE*)&currentBlockIdx, 8);
	}
}

void pre_sha256() {
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}

// Prints a 32 bytes sha256 to the hexadecimal form filled with zeroes
void print_hash(const unsigned char* sha256) {
	for (size_t i = 0; i < 32; ++i) {
		std::cout << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(sha256[i]);
	}
	std::cout << std::dec << std::endl;
}

bool isTimeToGetNewWork()
{
	std::chrono::high_resolution_clock::time_point tNow = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> lastWorkFetchInterval = tNow - t_last_work_fetch;
	if (lastWorkFetchInterval.count() > gElapsedMilliSecMax) {
		t_last_work_fetch = tNow;
		return true;
	}
	return false;
}

void print_state() {
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;
	if (last_show_interval.count() > SHOW_INTERVAL_MS) {
		t_last_updated = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> span = t2 - t1;
		float ratio = span.count() / 1000;
		//std::cout << std::fixed << static_cast<uint64_t>(totalNonce / ratio) << " h/s S:" << totalSharesFound << " S/H:" << ((totalSharesFound *3600) / ratio) << std::endl;
		std::cout << std::fixed << static_cast<uint64_t>(totalNonce / ratio) << " h/s " << endl;
	}
}

void PrintWorkData(MinerThreadData * pThreadData)
{
	std::cout << "New Work ||" << "BDiff:" << pThreadData->blockDifficulty << " SDiff:" << pThreadData->shareDifficulty << endl;
}


#define SHA_PER_ITERATIONS 8'388'608
int gBlockSize = 192;
int gNumBlocks = 65536;//(SHA_PER_ITERATIONS + gBlockSize - 1) / gBlockSize;

int main(int argc, char* argv[])
{
	cout << TOOL_NAME << " v" << VERSION_STRING << endl;
	if (argc <= 1) {
		cout << "Example Template: " << endl;
		cout << argv[0] << " " << "<Base64AmoveoAddress>" << " " << "<CudaDeviceId>" << " " << "<BlockSize>" << " " << "<NumBlocks>" << " " << "<PoolUrl>" << endl;

		cout << endl;
		cout << "Example Usage: " << endl;
		cout << argv[0] << " " << MINER_ADDRESS << endl;

		cout << endl;
		cout << "Advanced Example Usage: " << endl;
		cout << argv[0] << " " << MINER_ADDRESS << " " << DEFAULT_DEVICE_ID << " " << gBlockSize << " " << gNumBlocks << " " << POOL_URL << endl;

		cout << endl;
		cout << endl;
		cout << "CudaDeviceId is optional. Default CudaDeviceId is 0" << endl;
		cout << "BlockSize is optional. Default BlockSize is 192" << endl;
		cout << "NumBlocks is optional. Default NumBlocks is 65536" << endl;
		cout << "PoolUrl is optional. Default PoolUrl is http://amoveopool.com/work" << endl;
		return -1;
	}
	if (argc >= 2) {
		gMinerPublicKeyBase64 = argv[1];
	}
	if (argc >= 3) {
		gDevicdeId = atoi(argv[2]);
	}
	if (argc >= 4) {
		gBlockSize = atoi(argv[3]);
	}
	if (argc >= 5) {
		gNumBlocks = atoi(argv[4]);
	}
	if (argc >= 6) {
		gPoolUrl = argv[5];
	}

	gPoolUrlW.resize(gPoolUrl.length(), L' ');
	std::copy(gPoolUrl.begin(), gPoolUrl.end(), gPoolUrlW.begin());

	// obtain a seed from a user string:
	std::string str;
	std::cout << "Please, enter a seed string (smash keys, then press enter): ";
	std::getline(std::cin, str);
	std::seed_seq seed1(str.begin(), str.end());

	PoolApi poolApi;
	std::independent_bits_engine<std::default_random_engine, 32, uint32_t> randomBytesEngine(seed1);

	cudaSetDevice(gDevicdeId);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, gDevicdeId);

	cout << "GPU Device Properties:" << endl;
	cout << "maxThreadsDim: " << deviceProp.maxThreadsDim << endl;
	cout << "maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << endl;
	cout << "maxGridSize: " << deviceProp.maxGridSize << endl;


	// Input string for the device
	unsigned char *d_in = nullptr;

	cudaMalloc(&d_in, 32); // bhash - fixed 32
	cudaMallocManaged(&g_out, 32); // nonce output - fixed 32
	cudaMallocManaged(&g_hash_out, 32); // sha hash output - fixed 32
	cudaMallocManaged(&g_found, sizeof(int)); // "found" success flag
	cudaMalloc(&g_nonce, 32);

	t1 = std::chrono::high_resolution_clock::now();
	t_last_updated = std::chrono::high_resolution_clock::now();
	t_last_work_fetch = std::chrono::high_resolution_clock::now();

	MinerThreadData threadData;
	threadData.nonce = vector<unsigned char>(24);
	std::generate(begin(threadData.nonce), end(threadData.nonce), std::ref(randomBytesEngine));

	poolApi.GetWork(gPoolUrlW, &threadData, gMinerPublicKeyBase64);
	PrintWorkData(&threadData);

	// Assuming bhash and nonce are fixed size, so dynamic_shared_size should never change across work units
	size_t dynamic_shared_size = threadData.bhash.size() + threadData.nonce.size() + (64 * gBlockSize);
	std::cout << "Shared memory is " << dynamic_shared_size << "B" << std::endl;

	const int blocksPerKernel = gNumBlocks * gBlockSize;

	cout << "blockSize: " << gBlockSize << endl;
	cout << "numBlocks: " << gNumBlocks << endl;
	
	while (true)
	{
		*g_found = 0;
		cudaMemcpy(d_in, &threadData.bhash[0], threadData.bhash.size(), cudaMemcpyHostToDevice);
		cudaMemcpy(g_nonce, &threadData.nonce[0], 24, cudaMemcpyHostToDevice);

		pre_sha256();

		while(true) {
			sha256_kernel <<< gNumBlocks, gBlockSize, dynamic_shared_size >>> (g_out, g_found, d_in, threadData.bhash.size(), threadData.blockDifficulty, threadData.shareDifficulty, g_nonce);

			cudaError_t err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cout << "Cuda Error: " << cudaGetErrorString(err) << std::endl;
				throw std::runtime_error("Device error");
			}

			totalNonce += blocksPerKernel;
			
			std::generate(begin(threadData.nonce), end(threadData.nonce), std::ref(randomBytesEngine));
			cudaMemcpy(g_nonce, &threadData.nonce[0], 24, cudaMemcpyHostToDevice);

			print_state();

			if (*g_found) {
				poolApi.SubmitWork(gPoolUrlW, base64_encode(g_out, 32), gMinerPublicKeyBase64);
				totalSharesFound++;
				cout << "--- Found Share --- SDiff:" << threadData.shareDifficulty << endl;
				//print_hash(&threadData.nonce[0]);
				//print_hash(g_out);
			}
			if (*g_found || isTimeToGetNewWork()) {
				*g_found = 0;
				MinerThreadData threadDataNew;
				poolApi.GetWork(gPoolUrlW, &threadDataNew, gMinerPublicKeyBase64);
				// Check if new work unit is actually different than what we currently have
				if (memcmp(&threadDataNew.bhash[0], &threadData.bhash[0], 32) != 0) {
					threadData.bhash = threadDataNew.bhash;
					threadData.blockDifficulty = threadDataNew.blockDifficulty;
					threadData.shareDifficulty = threadDataNew.shareDifficulty;

					PrintWorkData(&threadData);
					gElapsedMilliSecMax = FETCH_WORK_INTERVAL_MS;
					break; // Only break if we got new work
				} else {
					gElapsedMilliSecMax += 3000;
					// Even if new work is not available, shareDiff will likely change. Need to adjust, else will get a "low diff share" error.
					threadData.shareDifficulty = threadDataNew.shareDifficulty;
				}
			}
		}

	}

	cudaFree(g_nonce);
	cudaFree(g_out);
	cudaFree(g_hash_out);
	cudaFree(g_found);

	cudaFree(d_in);

	cudaDeviceReset();

	return 0;
}
