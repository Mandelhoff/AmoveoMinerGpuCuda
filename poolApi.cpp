#include <cpprest/http_client.h>
#include <cpprest/filestream.h>
#include <cpprest/http_listener.h>              // HTTP server
#include <cpprest/json.h>                       // JSON library
#include <cpprest/uri.h>                        // URI library

#include "poolApi.h"
#include "base64.h"

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>

void mySleep(unsigned milliseconds)
{
	Sleep(milliseconds);
}
#else
#include <unistd.h>
#include <openssl/sha.h>

void mySleep(unsigned milliseconds)
{
	usleep(milliseconds * 1000); // takes microseconds
}
#endif

using namespace std;
using namespace std::chrono;

using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams
using namespace web::http::experimental::listener;          // HTTP server
using namespace web::json;                                  // JSON library

wstring successResponse = L"[-6,102,111,117,110,100,32,119,111,114,107]";

void PoolApi::SubmitWork(string_t poolUrl, string nonceBase64, string minerPublicKeyBase64)
{
	http_client client(poolUrl);
	http_request request(methods::POST);
	std::stringstream body;
	body << "[\"work\",\"" << nonceBase64 << "\",\"" << minerPublicKeyBase64 << "\"]";
	request.set_body(body.str());

	try
	{
		http_response response = client.request(request).get();
		if (response.status_code() == status_codes::OK)
		{
			// Response data comes in as application/octet-stream, so extract_json throws an exception
			// Need to use extract_vector and then convert to string and then to json
			std::vector<unsigned char> responseData = response.extract_vector().get();

			wstring responseString(responseData.begin(), responseData.end());

			if (successResponse.compare(responseString) != 0) {
				wcout << "Info: " << responseString << endl;
			}
		}
	}
	catch (...) {
		wcout << "ERROR: SubmitWork Exception..." << endl;
	}
	return;
}

void PoolApi::GetWork(string_t poolUrl, WorkData * pMinerThreadData, string minerPublicKeyBase64)
{
	bool success = false;
	do {
		try {
			http_client client(poolUrl);
			http_request request(methods::POST);
			std::stringstream body;
			body << "[\"mining_data\",\"" << minerPublicKeyBase64 << "\"]";
			request.set_body(body.str());
			//wcout << request.to_string();

			http_response response = client.request(request).get();
			if (response.status_code() == status_codes::OK)
			{
				// Response data comes in as application/octet-stream, so extract_json throws an exception
				// Need to use extract_vector and then convert to string and then to json
				std::vector<unsigned char> responseData = response.extract_vector().get();

				wstring responseString(responseData.begin(), responseData.end());

				json::value jsonResponse = json::value::parse(responseString);
				json::array dataArray = jsonResponse.as_array();

				wstring wBHhashBase64(dataArray[1].at(1).as_string().c_str());
				string bhashBase64(wBHhashBase64.begin(), wBHhashBase64.end());
				string bhashString = base64_decode(bhashBase64);
				vector<unsigned char> bhash(bhashString.begin(), bhashString.end());
				pMinerThreadData->bhash = bhash;

				int blockDifficulty = dataArray[1].at(2).as_integer();
				pMinerThreadData->blockDifficulty = blockDifficulty;

				int shareDifficulty = dataArray[1].at(3).as_integer();
				pMinerThreadData->shareDifficulty = shareDifficulty;

				success = true;
			}
			else {
				wcout << "ERROR: GetWork: " << response.status_code() << " Sleep and retry..." << endl;
				mySleep(3000);
			}
		}
		catch (...) {
			wcout << "ERROR: GetWork failed. Sleep and retry..." << endl;
			mySleep(3000);
		}
	} while (!success);

	return;
}

