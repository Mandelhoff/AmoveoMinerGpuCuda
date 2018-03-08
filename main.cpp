#define VERSION_STRING "1.0.0.4"
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
#include "stdlib.h"

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
#define SHOW_INTERVAL_MS 4000

int gElapsedMilliSecMax = FETCH_WORK_INTERVAL_MS;

//#define POOL_URL "http://localhost:32371/work"	// local pool
#define POOL_URL "http://amoveopool.com/work"
#define MINER_ADDRESS "BPA3r0XDT1V8W4sB14YKyuu/PgC6ujjYooVVzq1q1s5b6CAKeu9oLfmxlplcPd+34kfZ1qx+Dwe3EeoPu0SpzcI="
#define DEFAULT_DEVICE_ID 0

string gMinerPublicKeyBase64(MINER_ADDRESS);
string gPoolUrl(POOL_URL);
string_t gPoolUrlW;
int gDevicdeId = DEFAULT_DEVICE_ID;

PoolApi gPoolApi;
WorkData gWorkData;
std::mutex mutexWorkData;


uint64_t gTotalNonce = 0;

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

__global__ void sha256_kernel(unsigned char * out_nonce, int *out_found, const SHA256_CTX * in_ctx, uint64_t nonceOffset, int shareDiff)
{
	__shared__ SHA256_CTX ctxShared;
	__shared__ int diff;
	__shared__ uint64_t nonceOff;

	// If this is the first thread of the block, init the input string in shared memory
	if (threadIdx.x == 0) {
		memcpy(&ctxShared, in_ctx, 0x70);
		diff = shareDiff;
		nonceOff = nonceOffset;
	}
	__syncthreads(); // Ensure the input string has been written in SMEM

	uint64_t currentBlockIdx = blockIdx.x * blockDim.x + threadIdx.x + nonceOff;

	unsigned char shaResult[32];
	SHA256_CTX ctx;
	memcpy(&ctx, &ctxShared, 0x70);

	sha256_update(&ctx, (BYTE*)&currentBlockIdx, 8);
	sha256_final(&ctx, shaResult);

	if (checkResult(shaResult, diff) && atomicExch(out_found, 1) == 0) {
		memcpy(out_nonce, &currentBlockIdx, 8);
	}
}

__global__ void sha256Init_kernel(unsigned char * out_ctx, unsigned char * bhash, unsigned char * noncePart, int diff)
{
	SHA256_CTX ctx;
	unsigned char bhashLocal[32];
	unsigned char nonceLocal[24];

	memcpy(bhashLocal, bhash, 32);
	memcpy(nonceLocal, noncePart, 24);

	unsigned char diffA = diff / 256;
	unsigned char diffB = diff % 256;

	sha256_init(&ctx);
	sha256_update(&ctx, bhashLocal, 32);
	sha256_update(&ctx, &diffA, 1);
	sha256_update(&ctx, &diffB, 1);
	sha256_update(&ctx, nonceLocal, 24);
	memcpy(out_ctx, &ctx, 0x70);
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
		//std::cout << std::fixed << static_cast<uint64_t>(gTotalNonce / ratio) << " h/s S:" << totalSharesFound << " S/H:" << ((totalSharesFound *3600) / ratio) << std::endl;
		std::cout << std::fixed << static_cast<uint64_t>(gTotalNonce / ratio) << " h/s " << endl;
	}
}

static bool getwork_thread(std::seed_seq &seed)
{
	std::independent_bits_engine<std::default_random_engine, 32, uint32_t> randomBytesEngine(seed);

	unsigned char ctxBuf[0x70];

	unsigned char *d_bhash = nullptr;
	unsigned char *d_nonce = nullptr;
	cudaMalloc(&d_bhash, 32);
	cudaMalloc(&d_nonce, 24);
	unsigned char * outCtx = nullptr;
	cudaMalloc(&outCtx, 0x70);

	while (true)
	{
		WorkData workDataNew;
		gPoolApi.GetWork(gPoolUrlW, &workDataNew, gMinerPublicKeyBase64);

		// Check if new work unit is actually different than what we currently have
		if (memcmp(&gWorkData.bhash[0], &workDataNew.bhash[0], 32) != 0) {
			mutexWorkData.lock();
			std::generate(begin(gWorkData.nonce), end(gWorkData.nonce), std::ref(randomBytesEngine));
			gWorkData.bhash = workDataNew.bhash;
			gWorkData.blockDifficulty = workDataNew.blockDifficulty;
			gWorkData.shareDifficulty = workDataNew.shareDifficulty;

			cudaMemcpy(d_bhash, &gWorkData.bhash[0], 32, cudaMemcpyHostToDevice);
			cudaMemcpy(d_nonce, &gWorkData.nonce[0], 24, cudaMemcpyHostToDevice);

			sha256Init_kernel <<< 1, 1 >>> (outCtx, d_bhash, d_nonce, gWorkData.blockDifficulty);

			cudaError_t err = cudaDeviceSynchronize();
			if (err != cudaSuccess) {
				std::cout << "getwork_thread Cuda Error: " << cudaGetErrorString(err) << std::endl;
				throw std::runtime_error("getwork_thread Device error");
			}

			cudaMemcpy(ctxBuf, outCtx, 0x70, cudaMemcpyDeviceToHost);
			//SHA256_CTX ctx;
			//memcpy(&ctx, outCtx, sizeof(SHA256_CTX));
			gWorkData.setCtx(ctxBuf);

			mutexWorkData.unlock();

			std::cout << "New Work ||" << "BDiff:" << gWorkData.blockDifficulty << " SDiff:" << gWorkData.shareDifficulty << endl;
		} else {
			// Even if new work is not available, shareDiff will likely change. Need to adjust, else will get a "low diff share" error.
			gWorkData.shareDifficulty = workDataNew.shareDifficulty;
		}

		_sleep(2000);
	}

	cudaFree(outCtx);
	cudaFree(d_bhash);
	cudaFree(d_nonce);
	return true;
}

static void submitwork_thread(unsigned char * nonceSolution)
{
	gPoolApi.SubmitWork(gPoolUrlW, base64_encode(nonceSolution, 32), gMinerPublicKeyBase64);
	cout << "--- Found Share --- " << endl;
}

#define SHA_PER_ITERATIONS 8'388'608
int gBlockSize = 256;
int gNumBlocks = 65536;//(SHA_PER_ITERATIONS + gBlockSize - 1) / gBlockSize;
std::string gSeedStr("ImAraNdOmStrInG");

int main(int argc, char* argv[])
{
	cout << TOOL_NAME << " v" << VERSION_STRING << endl;
	if (argc <= 1) {
		cout << "Example Template: " << endl;
		cout << argv[0] << " " << "<Base64AmoveoAddress>" << " " << "<CudaDeviceId>" << " " << "<BlockSize>" << " " << "<NumBlocks>" << " " << "<SeedString>" << " " << "<PoolUrl>" << endl;

		cout << endl;
		cout << "Example Usage: " << endl;
		cout << argv[0] << " " << MINER_ADDRESS << endl;

		cout << endl;
		cout << "Advanced Example Usage: " << endl;
		cout << argv[0] << " " << MINER_ADDRESS << " " << DEFAULT_DEVICE_ID << " " << gBlockSize << " " << gNumBlocks << " " << "RandomSeed" << " " << POOL_URL << endl;

		cout << endl;
		cout << endl;
		cout << "CudaDeviceId is optional. Default CudaDeviceId is 0" << endl;
		cout << "BlockSize is optional. Default BlockSize is 256" << endl;
		cout << "NumBlocks is optional. Default NumBlocks is 65536" << endl;
		cout << "RandomSeed is option. No default." << endl;
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
		gSeedStr = argv[5];
	}
	if (argc >= 7) {
		gPoolUrl = argv[6];
	}
	cout << "gNumBlocks = " << gNumBlocks << endl;

	gPoolUrlW.resize(gPoolUrl.length(), L' ');
	std::copy(gPoolUrl.begin(), gPoolUrl.end(), gPoolUrlW.begin());
	std::seed_seq seed(gSeedStr.begin(), gSeedStr.end());

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, gDevicdeId);
	cout << "GPU Device Properties:" << endl;
	cout << "maxThreadsDim: " << deviceProp.maxThreadsDim << endl;
	cout << "maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << endl;
	cout << "maxGridSize: " << deviceProp.maxGridSize << endl;

	cudaSetDevice(gDevicdeId);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

	unsigned char localCtx[0x70];
	// Input string for the device
	SHA256_CTX * d_ctx = nullptr;
	// Output string by the device read by host
	unsigned char *g_out = nullptr;
	int *g_found = nullptr;

	cudaMalloc(&d_ctx, sizeof(SHA256_CTX)); // SHA256_CTX ctx to be used

	cudaMalloc(&g_out, 8); // partial nonce - last 8 bytes
	cudaMalloc(&g_found, 4); // "found" success flag

	future<bool> getWorkThread = std::async(std::launch::async, getwork_thread, std::ref(seed));

	// Assuming bhash and nonce are fixed size, so dynamic_shared_size should never change across work units
	size_t dynamic_shared_size = sizeof(SHA256_CTX) + sizeof(uint64_t) + sizeof(int);// +(64 * gBlockSize);
	std::cout << "Shared memory is " << dynamic_shared_size << "B" << std::endl;

	const int blocksPerKernel = gNumBlocks * gBlockSize;
	cout << "blockSize: " << gBlockSize << endl;
	cout << "numBlocks: " << gNumBlocks << endl;
	
	pre_sha256();

	uint64_t nonceOffset = 0;
	int shareDiff = 0;
	uint64_t nonceSolutionVal = 0;
	bool found = false;

	while (!gWorkData.HasNewWork())
	{
		_sleep(100);
	}
	gWorkData.getCtx(localCtx);
	cudaMemcpy(d_ctx, localCtx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice);
	int foundInit = 0;
	cudaMemcpy(g_found, &foundInit, 4, cudaMemcpyHostToDevice);
	gWorkData.clearNewWork();
	shareDiff = gWorkData.shareDifficulty;

	t1 = std::chrono::high_resolution_clock::now();
	t_last_updated = std::chrono::high_resolution_clock::now();
	t_last_work_fetch = std::chrono::high_resolution_clock::now();

	while(true) {

		sha256_kernel <<< gNumBlocks, gBlockSize, dynamic_shared_size >>> (g_out, g_found, d_ctx, nonceOffset, shareDiff);

		cudaError_t err = cudaDeviceSynchronize();
		if (err != cudaSuccess) {
			std::cout << "Cuda Error: " << cudaGetErrorString(err) << std::endl;
			throw std::runtime_error("Device error");
		}

		cudaMemcpy(&found, g_found, 1, cudaMemcpyDeviceToHost);

		if (found) {
			unsigned char nonceSolution[32];
			mutexWorkData.lock();
			memcpy(nonceSolution, &gWorkData.nonce[0], 24);
			mutexWorkData.unlock();
			cudaMemcpy(&nonceSolutionVal, g_out, 8, cudaMemcpyDeviceToHost);
			memcpy(nonceSolution + 24, &nonceSolutionVal, 8);
			//print_hash(nonceSolution);
			//print_hash(g_hash);
			//poolApi.SubmitWork(gPoolUrlW, base64_encode(nonceSolution, 32), gMinerPublicKeyBase64);

			std::async(std::launch::async, submitwork_thread, std::ref(nonceSolution));

			found = 0;
			cudaMemcpy(g_found, &found, 1, cudaMemcpyHostToDevice);
		}

		gTotalNonce += blocksPerKernel;
		nonceOffset += blocksPerKernel;

		if (gWorkData.HasNewWork())
		{
			mutexWorkData.lock();
			gWorkData.getCtx(localCtx);
			mutexWorkData.unlock();

			cudaMemcpy(d_ctx, localCtx, sizeof(SHA256_CTX), cudaMemcpyHostToDevice);
			gWorkData.clearNewWork();
			//nonceOffset = 0;
		}
		shareDiff = gWorkData.shareDifficulty;
		//print_state();

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> last_show_interval = t2 - t_last_updated;
		if (last_show_interval.count() > SHOW_INTERVAL_MS) {
			t_last_updated = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> span = t2 - t1;
			float ratio = span.count() / 1000;
			//std::cout << std::fixed << static_cast<uint64_t>(gTotalNonce / ratio) << " h/s S:" << totalSharesFound << " S/H:" << ((totalSharesFound *3600) / ratio) << std::endl;
			std::cout << std::fixed << static_cast<uint64_t>(gTotalNonce / ratio) << " h/s " << endl;
		}
	}

	cudaFree(g_out);
	cudaFree(g_found);

	cudaFree(d_ctx);

	cudaDeviceReset();

	return 0;
}

