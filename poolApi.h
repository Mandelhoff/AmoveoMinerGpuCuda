#pragma once


#include <string>
#include <vector>

using namespace std;
using namespace utility;                    // Common utilities like string conversions

void mySleep(unsigned milliseconds);

class WorkData
{
private:
	unsigned char ctx[0x70];
	bool newWork = false;

public:
	vector<unsigned char> bhash;
	vector<unsigned char> nonce;
	int blockDifficulty;
	int shareDifficulty;

	std::mutex mutexNewWork;
	//std::mutex mutexCtx;

public:
	WorkData() {
		bhash = vector<unsigned char>(32);
		nonce = vector<unsigned char>(15);
	}

	bool HasNewWork() { return newWork; }
	void clearNewWork()
	{
		mutexNewWork.lock();
		newWork = false;
		mutexNewWork.unlock();
	}

	void getCtx(unsigned char * pCtx) {
		//mutexCtx.lock();
		memcpy(pCtx, ctx, 0x70);
		//mutexCtx.unlock();
	}
	void setCtx(unsigned char * pCtx)
	{
		//mutexCtx.lock();
		memcpy(ctx, pCtx, 0x70);
		//mutexCtx.unlock();
		mutexNewWork.lock();
		newWork = true;
		mutexNewWork.unlock();
	}



};

class PoolApi
{
	public:
		void GetWork(string_t poolUrl, WorkData * pMinerThreadData, string minerPublicKeyBase64);
		void SubmitWork(string_t poolUrl, string nonceBase64, string minerPublicKeyBase64);


};
