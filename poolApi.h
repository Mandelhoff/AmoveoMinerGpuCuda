#pragma once

#include <string>
#include <vector>

using namespace std;
using namespace utility;                    // Common utilities like string conversions

struct MinerThreadData
{
	vector<unsigned char> bhash;
	vector<unsigned char> nonce;
	int blockDifficulty;
	int shareDifficulty;
};

class PoolApi
{
	public:
		void GetWork(string_t poolUrl, MinerThreadData * pMinerThreadData, string minerPublicKeyBase64);
		void SubmitWork(string_t poolUrl, string nonceBase64, string minerPublicKeyBase64);


};